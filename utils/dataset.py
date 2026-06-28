import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class FusionDataset(Dataset):
    def __init__(self, rgb_dir, voxel_dir, label_dir, img_size=640):
        self.rgb_dir   = Path(rgb_dir)
        self.voxel_dir = Path(voxel_dir)
        self.label_dir = Path(label_dir)
        self.img_size  = img_size
        self.stems = [
            p.stem for p in sorted(self.rgb_dir.glob('frame_*.png'))
            if (self.voxel_dir / f'{p.stem}.npy').exists() and
               (self.label_dir / f'{p.stem}.txt').exists()
        ]
        print(f'  {len(self.stems)} frames — {rgb_dir}')

    def __len__(self): return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # RGB
        rgb = cv2.cvtColor(
            cv2.imread(str(self.rgb_dir / f'{stem}.png')),
            cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2,0,1)

        # Voxel
        voxel = np.load(str(self.voxel_dir / f'{stem}.npy')).astype(np.float32)
        bins  = [cv2.resize(voxel[b], (self.img_size, self.img_size))
                 for b in range(voxel.shape[0])]
        voxel = torch.from_numpy(np.stack(bins))

        # Labels
        with open(self.label_dir / f'{stem}.txt') as f:
            lines = f.read().strip().splitlines()
        boxes = [list(map(float, l.split())) for l in lines if l]
        boxes = torch.tensor(boxes, dtype=torch.float32) \
                if boxes else torch.zeros((0,5))

        return {'rgb': rgb, 'voxel': voxel, 'labels': boxes, 'stem': stem}


def collate_fn(batch):
    return {
        'rgb':    torch.stack([b['rgb']   for b in batch]),
        'voxel':  torch.stack([b['voxel'] for b in batch]),
        'labels': [b['labels'] for b in batch],
        'stems':  [b['stem']   for b in batch],
    }


def get_dataloaders(cfg):
    train_ds = FusionDataset(cfg['train_rgb'], cfg['train_voxel'],
                             cfg['train_labels'], cfg['img_size'])
    val_ds   = FusionDataset(cfg['val_rgb'],   cfg['val_voxel'],
                             cfg['val_labels'],   cfg['img_size'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader
