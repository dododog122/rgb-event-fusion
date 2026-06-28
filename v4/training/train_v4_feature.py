import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics import YOLO
from v4.model.event_transformer import EventMAETransformer

DEVICE   = 'cuda'
BASE     = '/home/ivlab/carla_ws/fusion_model'
CKPT_DIR = Path(f'{BASE}/checkpoints')
IMG_SIZE = 640
NC       = 4


# ══════════════════════════════════════════
# Cross-Attention Fusion
# ══════════════════════════════════════════
class CrossAttentionFusion(nn.Module):
    def __init__(self, rgb_ch, event_dim=256, n_heads=4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = rgb_ch // n_heads
        self.scale    = self.head_dim ** -0.5
        self.q   = nn.Linear(rgb_ch,    rgb_ch)
        self.k   = nn.Linear(event_dim, rgb_ch)
        self.v   = nn.Linear(event_dim, rgb_ch)
        self.out = nn.Linear(rgb_ch,    rgb_ch)
        self.norm = nn.LayerNorm(rgb_ch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(rgb_ch, 1), nn.Sigmoid())

    def forward(self, rgb_feat, evt_tokens):
        B, C, H, W = rgb_feat.shape
        N  = evt_tokens.shape[1]
        nh = self.n_heads
        hd = self.head_dim
        x    = rgb_feat.flatten(2).transpose(1, 2)       # (B,HW,C)
        gate = self.gate(rgb_feat).unsqueeze(1)           # (B,1,1)
        Q = self.q(x).view(B,H*W,nh,hd).transpose(1,2)
        K = self.k(evt_tokens).view(B,N,nh,hd).transpose(1,2)
        V = self.v(evt_tokens).view(B,N,nh,hd).transpose(1,2)
        attn = (Q @ K.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ V).transpose(1,2).reshape(B, H*W, C)
        out  = self.out(out)
        fused = self.norm(x + out * gate)
        return fused.transpose(1,2).reshape(B, C, H, W)


# ══════════════════════════════════════════
# V4 Model — hardcoded forward
# ══════════════════════════════════════════
class V4FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_enc = EventMAETransformer(
            in_ch=5, img_size=640, patch_size=16,
            embed_dim=256, depth=6, n_heads=8, mask_ratio=0.75)

        yolo = YOLO('yolov8n.pt')
        m = yolo.model.model
        # store all 23 layers individually
        self.l0  = m[0];  self.l1  = m[1];  self.l2  = m[2]
        self.l3  = m[3];  self.l4  = m[4];  self.l5  = m[5]
        self.l6  = m[6];  self.l7  = m[7];  self.l8  = m[8]
        self.l9  = m[9];  self.l10 = m[10]; self.l11 = m[11]
        self.l12 = m[12]; self.l13 = m[13]; self.l14 = m[14]
        self.l15 = m[15]; self.l16 = m[16]; self.l17 = m[17]
        self.l18 = m[18]; self.l19 = m[19]; self.l20 = m[20]
        self.l21 = m[21]; self.l22 = m[22]

        # fusion at P3(layer4,ch=64) P4(layer6,ch=128)
        self.fuse_p3 = CrossAttentionFusion(64,  256, 4)
        self.fuse_p4 = CrossAttentionFusion(128, 256, 4)

    def forward(self, rgb, voxel):
        # event tokens
        evt = self.event_enc(voxel, pretrain=False)  # (B,1600,256)

        # backbone
        x0  = self.l0(rgb)
        x1  = self.l1(x0)
        x2  = self.l2(x1)
        x3  = self.l3(x2)
        x4  = self.l4(x3)
        x5  = self.l5(x4)
        x6  = self.l6(x5)
        x7  = self.l7(x6)
        x8  = self.l8(x7)
        x9  = self.l9(x8)

        # inject fusion at P3 and P4
        x4f = self.fuse_p3(x4, evt)   # fused P3 (B,64,80,80)
        x6f = self.fuse_p4(x6, evt)   # fused P4 (B,128,40,40)

        # neck
        x10 = self.l10(x9)
        x11 = self.l11([x10, x6f])   # Concat[-1, 6]
        x12 = self.l12(x11)
        x13 = self.l13(x12)
        x14 = self.l14([x13, x4f])   # Concat[-1, 4]
        x15 = self.l15(x14)
        x16 = self.l16(x15)
        x17 = self.l17([x16, x12])   # Concat[-1, 12]
        x18 = self.l18(x17)
        x19 = self.l19(x18)
        x20 = self.l20([x19, x9])    # Concat[-1, 9]
        x21 = self.l21(x20)

        # detect
        out = self.l22([x15, x18, x21])
        return out


# ══════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════
class V4Dataset(Dataset):
    def __init__(self, rgb_dir, voxel_dir, label_dir):
        self.vd = Path(voxel_dir)
        self.ld = Path(label_dir)
        self.files = sorted([
            f for f in Path(rgb_dir).glob('frame_*.png')
            if (self.vd/f'{f.stem}.npy').exists()
            and (self.ld/f'{f.stem}.txt').exists()])
        print(f'  {len(self.files)} frames')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx].stem
        S    = IMG_SIZE
        rgb  = cv2.imread(str(self.files[idx]))
        rgb  = cv2.resize(rgb, (S,S))
        rgb  = torch.from_numpy(rgb).permute(2,0,1).float()/255.
        v    = np.load(str(self.vd/f'{stem}.npy')).astype(np.float32)
        vox  = torch.from_numpy(
            np.stack([cv2.resize(v[b],(S,S)) for b in range(5)]))
        boxes, cls_ids = [], []
        with open(self.ld/f'{stem}.txt') as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    c,x,y,w,h = map(float,p)
                    cls_ids.append(int(c))
                    boxes.append([x,y,w,h])
        boxes   = torch.tensor(boxes,   dtype=torch.float32) \
                  if boxes   else torch.zeros((0,4))
        cls_ids = torch.tensor(cls_ids, dtype=torch.long) \
                  if cls_ids else torch.zeros((0,), dtype=torch.long)
        return rgb, vox, boxes, cls_ids

def collate_fn(batch):
    r,v,b,c = zip(*batch)
    return torch.stack(r), torch.stack(v), list(b), list(c)


# ══════════════════════════════════════════
# Loss
# ══════════════════════════════════════════
def compute_loss(pred_out, gt_boxes_list, gt_cls_list, device='cuda'):
    """
    Training mode output is dict:
      boxes:  (B, 64, 8400)  — box regression (DFL format)
      scores: (B, 80, 8400)  — class scores (80 COCO classes)
      feats:  list of 3 feature maps
    We use scores[:, :NC, :] for our 4 classes.
    """
    if isinstance(pred_out, dict):
        pred_box = pred_out["boxes"].permute(0,2,1)    # (B,8400,64)
        pred_cls = pred_out["scores"].permute(0,2,1)   # (B,8400,80)
    elif isinstance(pred_out, (list,tuple)):
        # find dict
        for p in pred_out:
            if isinstance(p, dict):
                pred_box = p["boxes"].permute(0,2,1)
                pred_cls = p["scores"].permute(0,2,1)
                break
    else:
        raise ValueError(f"Unexpected pred type: {type(pred_out)}")

    B  = pred_cls.shape[0]
    N  = pred_cls.shape[1]    # 8400
    pred_cls_nc = pred_cls[:, :, :NC]  # (B,8400,NC) use first NC classes

    # anchor grid
    anchors = []
    for sz in [80, 40, 20]:
        gy, gx = torch.meshgrid(
            torch.arange(sz, dtype=torch.float32, device=device),
            torch.arange(sz, dtype=torch.float32, device=device),
            indexing="ij")
        grid = torch.stack([(gx+0.5)/sz, (gy+0.5)/sz], dim=-1)
        anchors.append(grid.reshape(-1,2))
    anchors = torch.cat(anchors)  # (8400,2)

    all_losses = []

    for b in range(B):
        gt_b = gt_boxes_list[b].to(device)  # (M,4) xywh normalized
        cl_b = gt_cls_list[b].to(device)    # (M,)

        obj_tgt = torch.zeros(N, device=device)

        if gt_b.shape[0] == 0:
            obj_l = F.binary_cross_entropy_with_logits(
                pred_cls_nc[b].max(-1).values, obj_tgt)
            all_losses.append(obj_l)
            continue

        # assign top-10 nearest anchors per GT
        dist   = torch.cdist(anchors, gt_b[:,:2])        # (8400,M)
        _, topk = dist.topk(10, dim=0, largest=False)    # (10,M)

        # build targets
        pos_sets = [topk[:,n] for n in range(gt_b.shape[0])]
        pos_idx  = torch.cat(pos_sets).unique()
        obj_tgt[pos_idx] = 1.

        # obj loss
        obj_l = F.binary_cross_entropy_with_logits(
            pred_cls_nc[b].max(-1).values, obj_tgt)

        # box loss — use xy prediction (first 4 of 64 DFL channels)
        pb = pred_box[b][pos_idx][:, :4].sigmoid()  # (K,4)
        # build box targets for each pos anchor
        box_tgt = torch.zeros(len(pos_idx), 4, device=device)
        cls_tgt = torch.zeros(len(pos_idx), NC, device=device)

        for n in range(gt_b.shape[0]):
            pidx_n = topk[:,n]
            mask   = torch.isin(pos_idx, pidx_n)
            box_tgt[mask] = gt_b[n]
            cls_tgt[mask, cl_b[n]] = 1.

        box_l = F.mse_loss(pb, box_tgt)
        cls_l = F.binary_cross_entropy_with_logits(
            pred_cls_nc[b][pos_idx], cls_tgt)

        all_losses.append(obj_l + box_l + cls_l)

    return torch.stack(all_losses).mean()



# ══════════════════════════════════════════
# Train
# ══════════════════════════════════════════
def train(epochs=50):
    print('\n=== V4 True Feature-level Fusion ===')

    train_ds = V4Dataset(f'{BASE}/morning/rgb',
                         f'{BASE}/morning/voxel',
                         f'{BASE}/morning/yolo/labels')
    val_ds   = V4Dataset(f'{BASE}/night/rgb',
                         f'{BASE}/night/voxel',
                         f'{BASE}/night/yolo/labels')
    train_ld = DataLoader(train_ds, batch_size=2, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

    model = V4FeatureModel().to(DEVICE)
    model.event_enc.load_state_dict(
        torch.load(CKPT_DIR/'v4_event_encoder.pt', map_location=DEVICE))
    print('Event encoder loaded!')

    # freeze encoder initially
    for p in model.event_enc.parameters():
        p.requires_grad = False

    trainable = (list(model.fuse_p3.parameters()) +
                 list(model.fuse_p4.parameters()) +
                 [p for name, p in model.named_parameters()
                  if name.startswith('l') and p.requires_grad])

    opt = AdamW(trainable, lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    best = float('inf')

    for ep in range(1, epochs+1):
        if ep == 15:
            print('Unfreezing event encoder...')
            for p in model.event_enc.parameters():
                p.requires_grad = True
            opt.add_param_group(
                {'params': model.event_enc.parameters(), 'lr':1e-5})

        model.train()
        tr = 0
        for rgb, vox, boxes, cls_ids in tqdm(
                train_ld, desc=f'Ep{ep:02d}/{epochs}'):
            rgb = rgb.to(DEVICE)
            vox = vox.to(DEVICE)
            opt.zero_grad()
            pred = model(rgb, vox)
            loss = compute_loss(pred, boxes, cls_ids, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tr += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for rgb, vox, boxes, cls_ids in val_ld:
                rgb = rgb.to(DEVICE)
                vox = vox.to(DEVICE)
                pred = model(rgb, vox)
                loss = compute_loss(pred, boxes, cls_ids, DEVICE)
                vl += loss.item()

        sch.step()
        tl = tr / len(train_ld)
        vl = vl / len(val_ld)
        print(f'Ep {ep:02d} | train={tl:.4f}  val={vl:.4f}')

        if vl < best:
            best = vl
            torch.save(model.state_dict(),
                       CKPT_DIR/'v4_feature_best.pt')
            print(f'  Saved! val={vl:.4f}')

    print(f'\nDone! Best val={best:.4f}')


if __name__ == '__main__':
    train(epochs=50)
