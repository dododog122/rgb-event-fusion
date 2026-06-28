import sys
import os
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

from v4.model.event_transformer import EventMAETransformer
from v4.model.fusion import CrossModalCBAM
from utils.dataset import FusionDataset

DEVICE   = 'cuda'
BASE     = '/home/ivlab/carla_ws/fusion_model'
CKPT_DIR = Path(f'{BASE}/checkpoints')

# ── Dataset ──
class V4Dataset(torch.utils.data.Dataset):
    def __init__(self, rgb_dir, voxel_dir, label_dir, img_size=640):
        self.img_size  = img_size
        self.rgb_files = sorted(Path(rgb_dir).glob('frame_*.png'))
        self.voxel_dir = Path(voxel_dir)
        self.label_dir = Path(label_dir)
        # filter valid
        self.files = [f for f in self.rgb_files
                      if (self.voxel_dir/f'{f.stem}.npy').exists()
                      and (self.label_dir/f'{f.stem}.txt').exists()]
        print(f'  {len(self.files)} valid frames')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx].stem
        S    = self.img_size

        # RGB
        rgb = cv2.imread(str(self.files[idx]))
        rgb = cv2.resize(rgb, (S, S))
        rgb = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0

        # Voxel
        v    = np.load(str(self.voxel_dir/f'{stem}.npy')).astype(np.float32)
        bins = [cv2.resize(v[b], (S,S)) for b in range(v.shape[0])]
        voxel = torch.from_numpy(np.stack(bins))

        # Labels
        boxes, cls_ids = [], []
        with open(self.label_dir/f'{stem}.txt') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, x, y, w, h = map(float, parts)
                    cls_ids.append(int(c))
                    boxes.append([x, y, w, h])

        boxes   = torch.tensor(boxes,   dtype=torch.float32) if boxes   else torch.zeros((0,4))
        cls_ids = torch.tensor(cls_ids, dtype=torch.long)    if cls_ids else torch.zeros((0,), dtype=torch.long)

        return rgb, voxel, boxes, cls_ids


def collate_fn(batch):
    rgbs, voxels, boxes, cls_ids = zip(*batch)
    return (torch.stack(rgbs),
            torch.stack(voxels),
            list(boxes),
            list(cls_ids))


# ── Detection Head ──
class DetectionHead(nn.Module):
    """Simple single-scale detection head on top of fused P3 features."""
    def __init__(self, in_ch=128, num_classes=4, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out = num_anchors * (5 + num_classes)  # 5 = x,y,w,h,obj
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch*2),
            nn.SiLU(),
            nn.Conv2d(in_ch*2, out, 1),
        )

    def forward(self, x):
        return self.head(x)


# ── Full V4 model ──
class NightFusionV4Full(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        # Event encoder (load pretrained)
        self.event_encoder = EventMAETransformer(
            in_ch=5, img_size=640, patch_size=16,
            embed_dim=256, depth=6, n_heads=8, mask_ratio=0.75
        )

        # RGB backbone — use YOLOv8 CSPDarknet layers
        from ultralytics import YOLO
        yolo = YOLO('yolov8n.pt')
        self.yolo_layers = yolo.model.model  # all layers

        # Fusion modules
        self.fusion_p3 = CrossModalCBAM(rgb_ch=64, event_dim=256, out_ch=64)
        self.fusion_p4 = CrossModalCBAM(rgb_ch=128, event_dim=256, out_ch=128)

        # Detection head (on P3)
        self.det_head = DetectionHead(in_ch=64, num_classes=num_classes)

    def get_backbone_features(self, rgb):
        x = rgb
        p3 = p4 = None
        for i, layer in enumerate(self.yolo_layers):
            x = layer(x)
            if i == 4:  p3 = x   # (B,64,80,80)   # (B,128,80,80)
            if i == 6:  p4 = x   # (B,128,40,40)   # (B,256,40,40)
            if i >= 6:  break
        return p3, p4

    def forward(self, rgb, voxel):
        # Event tokens
        event_tokens = self.event_encoder(voxel, pretrain=False)

        # RGB features
        p3, p4 = self.get_backbone_features(rgb)

        # Feature-level fusion
        f3 = self.fusion_p3(p3, event_tokens, rgb)   # (B,64,80,80)
        f4 = self.fusion_p4(p4, event_tokens, rgb)   # (B,128,40,40)

        # Detection on fused P3
        pred = self.det_head(f3)  # (B, A*(5+C), 80, 80)
        return pred, f3, f4


# ── Simple detection loss ──
def detection_loss(pred, gt_boxes_list, gt_cls_list, num_classes=4, device='cuda'):
    """
    Simple anchor-free detection loss.
    pred: (B, A*(5+C), H, W)
    """
    B, _, H, W = pred.shape
    A = 3  # anchors
    C = num_classes

    pred = pred.view(B, A, 5+C, H, W).permute(0,1,3,4,2)
    # pred: (B, A, H, W, 5+C)

    pred_xy  = pred[..., :2].sigmoid()
    pred_wh  = pred[..., 2:4]
    pred_obj = pred[..., 4]
    pred_cls = pred[..., 5:]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_pos = 0

    for b in range(B):
        gt_boxes = gt_boxes_list[b].to(device)  # (N, 4) xywh normalized
        gt_cls   = gt_cls_list[b].to(device)    # (N,)

        if gt_boxes.shape[0] == 0:
            # no GT: objectness should all be 0
            obj_loss = F.binary_cross_entropy_with_logits(
                pred_obj[b], torch.zeros_like(pred_obj[b]))
            total_loss = total_loss + obj_loss
            continue

        # assign GT to grid cells
        gt_cx = (gt_boxes[:, 0] * W).long().clamp(0, W-1)
        gt_cy = (gt_boxes[:, 1] * H).long().clamp(0, H-1)

        obj_target = torch.zeros(A, H, W, device=device)
        box_loss   = torch.tensor(0.0, device=device)
        cls_loss   = torch.tensor(0.0, device=device)

        for n in range(gt_boxes.shape[0]):
            cx, cy = gt_cx[n], gt_cy[n]
            for a in range(A):
                obj_target[a, cy, cx] = 1.0

                # box loss (L1 on xy, wh)
                pred_box = torch.cat([
                    pred_xy[b, a, cy, cx],
                    pred_wh[b, a, cy, cx]
                ])
                gt_box = gt_boxes[n].to(device)
                box_loss = box_loss + F.mse_loss(pred_box, gt_box)

                # cls loss
                cls_t = torch.zeros(C, device=device)
                cls_t[gt_cls[n]] = 1.0
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    pred_cls[b, a, cy, cx], cls_t)
                n_pos += 1

        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj[b], obj_target)

        if n_pos > 0:
            total_loss = total_loss + obj_loss + \
                         box_loss / n_pos + cls_loss / n_pos
        else:
            total_loss = total_loss + obj_loss

    return total_loss / B


# ── Training ──
def train(epochs=50):
    print(f'\n=== V4 Feature-level Detection Training ===')
    print(f'Device: {DEVICE}')

    # Dataset
    train_ds = V4Dataset(
        f'{BASE}/morning/rgb',
        f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels',
    )
    val_ds = V4Dataset(
        f'{BASE}/night/rgb',
        f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels',
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # Model
    model = NightFusionV4Full(num_classes=4).to(DEVICE)

    # Load pretrained event encoder
    ckpt = torch.load(CKPT_DIR/'v4_event_encoder.pt', map_location=DEVICE)
    model.event_encoder.load_state_dict(ckpt)
    print('Pretrained event encoder loaded!')

    # Freeze event encoder initially, only train fusion + head
    for p in model.event_encoder.parameters():
        p.requires_grad = False
    for p in model.yolo_layers.parameters():
        p.requires_grad = False

    trainable = list(model.fusion_p3.parameters()) + \
                list(model.fusion_p4.parameters()) + \
                list(model.det_head.parameters())

    optimizer = AdamW(trainable, lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(1, epochs+1):

        # Unfreeze event encoder after epoch 10
        if epoch == 10:
            print('Unfreezing event encoder...')
            for p in model.event_encoder.parameters():
                p.requires_grad = True
            optimizer.add_param_group(
                {'params': model.event_encoder.parameters(), 'lr': 1e-5})

        model.train()
        total = 0
        for rgb, voxel, boxes, cls_ids in tqdm(
                train_loader, desc=f'Epoch {epoch:02d}/{epochs}'):
            rgb   = rgb.to(DEVICE)
            voxel = voxel.to(DEVICE)

            optimizer.zero_grad()
            pred, f3, f4 = model(rgb, voxel)
            loss = detection_loss(pred, boxes, cls_ids,
                                  num_classes=4, device=DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss.item()

        # Validation
        model.eval()
        val_total = 0
        with torch.no_grad():
            for rgb, voxel, boxes, cls_ids in val_loader:
                rgb   = rgb.to(DEVICE)
                voxel = voxel.to(DEVICE)
                pred, _, _ = model(rgb, voxel)
                loss = detection_loss(pred, boxes, cls_ids,
                                      num_classes=4, device=DEVICE)
                val_total += loss.item()

        scheduler.step()
        tl = total     / len(train_loader)
        vl = val_total / len(val_loader)
        print(f'Epoch {epoch:02d} | train={tl:.4f}  val={vl:.4f}')

        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(),
                       CKPT_DIR/'v4_detection_best.pt')
            print(f'  Saved! (val={vl:.4f})')

    print(f'\nDone! Best val loss: {best_loss:.4f}')
    print('Next: run mAP evaluation with saved checkpoint')


if __name__ == '__main__':
    train(epochs=50)
