import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

from model.model    import NightFusionModel
from utils.dataset  import get_dataloaders
from utils.loss     import rsgnet_loss, detection_loss

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
    'batch_size':   2,
}
EPOCHS     = 30
LR         = 1e-4
LAMBDA_RSG = 0.5
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR   = Path('checkpoints')
CKPT_DIR.mkdir(exist_ok=True)
# ────────────────────────────────────────────


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total = 0
    ctx   = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc='Train' if train else 'Val'):
            rgb    = batch['rgb'].to(device)
            voxel  = batch['voxel'].to(device)
            labels = batch['labels']

            if train: optimizer.zero_grad()

            p3, p4, sif = model(rgb, voxel)
            l_rsg = rsgnet_loss(sif, voxel)
            l_det = detection_loss(p3, labels) + detection_loss(p4, labels)
            loss  = l_det + LAMBDA_RSG * l_rsg

            if train:
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def main():
    print(f'Device: {DEVICE}')
    train_loader, val_loader = get_dataloaders(CFG)
    model     = NightFusionModel(num_classes=4).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_loss = float('inf')

    for epoch in range(1, EPOCHS+1):
        t_loss = run_epoch(model, train_loader, optimizer, DEVICE, train=True)
        v_loss = run_epoch(model, val_loader,   optimizer, DEVICE, train=False)
        scheduler.step()
        print(f'Epoch {epoch:02d} | train={t_loss:.4f}  val={v_loss:.4f}')

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), CKPT_DIR/'best.pt')
            print(f'  ✅ Saved best  (val={v_loss:.4f})')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), CKPT_DIR/f'epoch_{epoch:02d}.pt')

    print(f'Done! Best val loss: {best_loss:.4f}')

if __name__ == '__main__':
    main()
