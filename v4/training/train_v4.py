import sys
import os
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

from v4.model.event_transformer import EventMAETransformer
from v4.utils.loss_v4 import FusionLossV4, patchify
from utils.dataset import get_dataloaders

BASE     = '/home/ivlab/carla_ws/fusion_model'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = Path(f'{BASE}/checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

CFG = {
    'train_rgb':    f'{BASE}/morning/rgb',
    'train_voxel':  f'{BASE}/morning/voxel',
    'train_labels': f'{BASE}/morning/yolo/labels',
    'val_rgb':      f'{BASE}/night/rgb',
    'val_voxel':    f'{BASE}/night/voxel',
    'val_labels':   f'{BASE}/night/yolo/labels',
    'img_size':     640,
    'batch_size':   4,
}

# ── Phase 1: Pre-train Event MAE Transformer ──────────────
def pretrain_event_mae(epochs=20):
    print('\n=== Phase 1: Pre-training Event MAE Transformer ===')
    train_loader, val_loader = get_dataloaders(CFG)

    model = EventMAETransformer(
        in_ch=5, img_size=640, patch_size=16,
        embed_dim=256, depth=6, n_heads=8,
        mask_ratio=0.75
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(1, epochs+1):
        model.train()
        total = 0
        for batch in tqdm(train_loader, desc=f'MAE Epoch {epoch}'):
            voxel = batch['voxel'].to(DEVICE)

            optimizer.zero_grad()
            pred, mask = model(voxel, pretrain=True)

            # reconstruction target: patchified original voxel
            target = patchify(voxel, patch_size=16)
            loss = ((pred - target) ** 2).mean(dim=-1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                voxel  = batch['voxel'].to(DEVICE)
                pred, mask = model(voxel, pretrain=True)
                target = patchify(voxel, patch_size=16)
                loss = ((pred - target)**2).mean(-1)
                loss = (loss*mask).sum()/(mask.sum()+1e-6)
                val_loss += loss.item()

        scheduler.step()
        tl = total / len(train_loader)
        vl = val_loss / len(val_loader)
        print(f'  Epoch {epoch:02d} | train={tl:.4f}  val={vl:.4f}')

        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(),
                       CKPT_DIR/'v4_mae_pretrain.pt')
            print(f'  Saved MAE checkpoint (val={vl:.4f})')

    print(f'MAE pre-training done! Best val loss: {best_loss:.4f}')
    return model


# ── Phase 2: Joint fusion training with YOLOv8 ────────────
def train_fusion(pretrained_mae, epochs=50):
    print('\n=== Phase 2: Joint Fusion Training ===')
    from ultralytics import YOLO

    # load pretrained event encoder weights
    print('Loading pretrained MAE weights...')

    # use YOLOv8 for detection, inject event features
    # simplest approach: train YOLOv8 on fusion images
    # + add auxiliary losses from event encoder

    train_loader, val_loader = get_dataloaders(CFG)

    event_encoder = EventMAETransformer(
        in_ch=5, img_size=640, patch_size=16,
        embed_dim=256, depth=6, n_heads=8,
        mask_ratio=0.75
    ).to(DEVICE)
    event_encoder.load_state_dict(
        torch.load(CKPT_DIR/'v4_mae_pretrain.pt',
                   map_location=DEVICE))

    aux_loss_fn = FusionLossV4(
        lambda_mae=0.3, lambda_align=0.2)

    # only fine-tune event encoder (backbone frozen initially)
    optimizer = AdamW(
        event_encoder.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(1, epochs+1):
        event_encoder.train()
        total = mae_t = align_t = 0

        for batch in tqdm(train_loader,
                          desc=f'Fusion Epoch {epoch}'):
            voxel = batch['voxel'].to(DEVICE)
            rgb   = batch['rgb'].to(DEVICE)

            optimizer.zero_grad()

            # event features (no mask in fusion mode)
            event_tokens = event_encoder(voxel, pretrain=False)

            # MAE aux loss (with mask)
            pred, mask = event_encoder(voxel, pretrain=True)
            target     = patchify(voxel, patch_size=16)

            # feature alignment: use rgb as proxy feature
            rgb_flat = rgb  # (B,3,H,W) — align_loss handles reshape

            total_loss, l_mae, l_align = aux_loss_fn(
                pred, target, mask,
                rgb_flat, event_tokens)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                event_encoder.parameters(), 1.0)
            optimizer.step()

            total   += total_loss.item()
            mae_t   += l_mae.item()
            align_t += l_align.item()

        scheduler.step()
        n = len(train_loader)
        print(f'Epoch {epoch:02d} | '
              f'total={total/n:.4f}  '
              f'mae={mae_t/n:.4f}  '
              f'align={align_t/n:.4f}')

        if total/n < best_loss:
            best_loss = total/n
            torch.save(event_encoder.state_dict(),
                       CKPT_DIR/'v4_event_encoder.pt')
            print(f'  Saved (loss={total/n:.4f})')

    print(f'\nPhase 2 done! Best loss: {best_loss:.4f}')
    print('Now train YOLOv8 with CLAHE+Event overlay')
    print('using: python3 -c "from ultralytics import YOLO; ...')


def main():
    # Skip Phase 1 - already done
    print(f'Device: {DEVICE}')
    # Phase 1: pre-train MAE
    pretrained = pretrain_event_mae(epochs=20)
    # Phase 2: joint training
    train_fusion(pretrained, epochs=30)


if __name__ == '__main__':
    main()
