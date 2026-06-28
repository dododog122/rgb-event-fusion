import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

from model.model         import NightFusionModel
from utils.dataset       import get_dataloaders
from utils.loss          import rsgnet_loss
from detection.utils.loss import YOLOLoss

# ── Config ──────────────────────────────────
BASE = '/home/ivlab/carla_ws/fusion_model'
CFG  = {
    'train_rgb':    f'{BASE}/morning/rgb',
    'train_voxel':  f'{BASE}/morning/voxel',
    'train_labels': f'{BASE}/morning/yolo/labels',
    'val_rgb':      f'{BASE}/night/rgb',
    'val_voxel':    f'{BASE}/night/voxel',
    'val_labels':   f'{BASE}/night/yolo/labels',
    'img_size':     640,
    'batch_size':   8,
}
EPOCHS     = 10
LR         = 1e-4
LAMBDA_RSG = 0.3
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR   = Path(BASE)/'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)
# ────────────────────────────────────────────


def run_epoch(model, loader, yolo_loss_fn,
              optimizer, device, train=True):
    model.train() if train else model.eval()
    total = box_t = obj_t = cls_t = rsg_t = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc='Train' if train else 'Val ')
        for batch in pbar:
            rgb    = batch['rgb'].to(device)
            voxel  = batch['voxel'].to(device)
            labels = batch['labels']

            if train:
                optimizer.zero_grad()

            p3, p4, sif = model(rgb, voxel)

            # Loss 1: RSGNet
            l_rsg = rsgnet_loss(sif, voxel)

            # Loss 2: YOLO detection
            l_det3, lb3, lo3, lc3 = yolo_loss_fn(p3, labels)
            l_det4, lb4, lo4, lc4 = yolo_loss_fn(p4, labels)
            l_det = l_det3 + l_det4

            loss = l_det + LAMBDA_RSG * l_rsg

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)
                optimizer.step()

            total += loss.item()
            box_t += (lb3+lb4).item()
            obj_t += (lo3+lo4).item()
            cls_t += (lc3+lc4).item()
            rsg_t += l_rsg.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box':  f'{(lb3+lb4).item():.4f}',
                'obj':  f'{(lo3+lo4).item():.4f}',
                'cls':  f'{(lc3+lc4).item():.4f}',
                'rsg':  f'{l_rsg.item():.4f}',
            })

    n = len(loader)
    return total/n, box_t/n, obj_t/n, cls_t/n, rsg_t/n


def main():
    print(f'Device: {DEVICE}')
    train_loader, val_loader = get_dataloaders(CFG)

    model         = NightFusionModel(num_classes=4).to(DEVICE)
    yolo_loss_fn  = YOLOLoss(num_classes=4, cls_weights=[2.0, 8.0, 15.0, 0.3])
    optimizer     = AdamW(model.parameters(),
                          lr=LR, weight_decay=1e-4)
    scheduler     = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_loss     = float('inf')

    print(f'Training for {EPOCHS} epochs...\n')

    for epoch in range(1, EPOCHS+1):
        tr = run_epoch(model, train_loader, yolo_loss_fn,
                       optimizer, DEVICE, train=True)
        va = run_epoch(model, val_loader,   yolo_loss_fn,
                       optimizer, DEVICE, train=False)
        scheduler.step()

        print(f'\nEpoch {epoch:02d}/{EPOCHS}')
        print(f'  Train — loss:{tr[0]:.4f}  box:{tr[1]:.4f}'
              f'  obj:{tr[2]:.4f}  cls:{tr[3]:.4f}  rsg:{tr[4]:.4f}')
        print(f'  Val   — loss:{va[0]:.4f}  box:{va[1]:.4f}'
              f'  obj:{va[2]:.4f}  cls:{va[3]:.4f}  rsg:{va[4]:.4f}')

        if va[0] < best_loss:
            best_loss = va[0]
            torch.save(model.state_dict(),
                       CKPT_DIR/'best.pt')
            print(f'  ✅ Saved best (val={va[0]:.4f})')

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       CKPT_DIR/f'epoch_{epoch:02d}.pt')

    print(f'\nDone! Best val loss: {best_loss:.4f}')

if __name__ == '__main__':
    main()
