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

DEVICE   = 'cuda'
BASE     = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'
CKPT_DIR = Path('/home/ivlab/carla_ws/fusion_model/checkpoints')
IMG_SIZE = 640
PATCH    = 16
N        = (IMG_SIZE // PATCH) ** 2  # 1600 per polarity
MASK_R   = 0.75


class PolarityAwareTokenizer(nn.Module):
    def __init__(self, in_ch=5, embed_dim=256):
        super().__init__()
        self.pos_temp = nn.Parameter(torch.ones(1) * 2.0)
        self.neg_temp = nn.Parameter(torch.ones(1) * 2.0)
        self.pos_embed = nn.Conv2d(in_ch, embed_dim, PATCH, PATCH, bias=False)
        self.neg_embed = nn.Conv2d(in_ch, embed_dim, PATCH, PATCH, bias=False)
        self.polarity_embed = nn.Embedding(2, embed_dim)
        self.norm_pos = nn.LayerNorm(embed_dim)
        self.norm_neg = nn.LayerNorm(embed_dim)

    def forward(self, voxel):
        pos = F.softplus(voxel *  self.pos_temp)
        neg = F.softplus(-voxel * self.neg_temp)
        pos = pos / (pos.flatten(2).max(-1)[0][...,None,None] + 1e-6)
        neg = neg / (neg.flatten(2).max(-1)[0][...,None,None] + 1e-6)
        dev = voxel.device
        pt = self.norm_pos(self.pos_embed(pos).flatten(2).transpose(1,2)
             + self.polarity_embed(torch.zeros(1,dtype=torch.long,device=dev)))
        nt = self.norm_neg(self.neg_embed(neg).flatten(2).transpose(1,2)
             + self.polarity_embed(torch.ones(1, dtype=torch.long,device=dev)))
        return torch.cat([pt, nt], dim=1)  # (B, 2N, D)


class ViTBlock(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.at = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        h,_ = self.at(self.n1(x), self.n1(x), self.n1(x))
        x = x + h
        return x + self.ff(self.n2(x))


class V5MAEModel(nn.Module):
    def __init__(self, embed_dim=256, depth=6):
        super().__init__()
        self.tok     = PolarityAwareTokenizer(5, embed_dim)
        self.pe      = nn.Parameter(torch.randn(1, N*2, embed_dim)*0.02)
        self.encoder = nn.ModuleList([ViTBlock(embed_dim) for _ in range(depth)])
        self.norm    = nn.LayerNorm(embed_dim)
        # lightweight decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.ModuleList([ViTBlock(embed_dim, heads=4) for _ in range(2)])
        self.dec_norm = nn.LayerNorm(embed_dim)
        self.pred    = nn.Linear(embed_dim, PATCH*PATCH*5)  # reconstruct voxel patches

    def patchify(self, voxel):
        """voxel (B,5,H,W) → patches (B, N, patch*patch*5)"""
        B,C,H,W = voxel.shape
        p = PATCH
        x = voxel.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0,2,4,1,3,5).reshape(B, (H//p)*(W//p), C*p*p)
        return x

    def forward(self, voxel):
        B = voxel.shape[0]
        tokens = self.tok(voxel) + self.pe  # (B, 2N, D)
        T = tokens.shape[1]

        # random mask
        noise = torch.rand(B, T, device=voxel.device)
        ids_s = noise.argsort(dim=1)
        keep  = int(T * (1 - MASK_R))
        ids_k = ids_s[:, :keep]
        ids_r = ids_s.argsort(dim=1)

        # encode visible
        vis = tokens.gather(1, ids_k.unsqueeze(-1).expand(-1,-1,tokens.shape[-1]))
        for blk in self.encoder:
            vis = blk(vis)
        vis = self.norm(vis)

        # decode with mask tokens
        mask_tok = self.mask_token.expand(B, T-keep, -1)
        full = torch.zeros(B, T, vis.shape[-1], device=voxel.device)
        full.scatter_(1, ids_k.unsqueeze(-1).expand(-1,-1,vis.shape[-1]), vis)
        full.scatter_(1, ids_s[:,keep:].unsqueeze(-1).expand(-1,-1,vis.shape[-1]), mask_tok)

        for blk in self.decoder:
            full = blk(full)
        full = self.dec_norm(full)

        # predict only on first N (positive polarity patches)
        pred = self.pred(full[:, :N, :])  # (B, N, patch*patch*5)

        # target
        target = self.patchify(voxel)  # (B, N, patch*patch*5)

        # loss only on masked patches
        mask = torch.ones(B, T, device=voxel.device)
        mask.scatter_(1, ids_k, 0)
        mask = mask[:, :N]  # only pos polarity side

        loss = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss


class VoxelDataset(Dataset):
    def __init__(self, voxel_dir):
        self.files = sorted(Path(voxel_dir).glob('frame_*.npy'))
        print(f'  {len(self.files)} voxel files')
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        v = np.load(str(self.files[idx])).astype(np.float32)
        return torch.from_numpy(
            np.stack([cv2.resize(v[b],(IMG_SIZE,IMG_SIZE)) for b in range(5)]))


def train(epochs=20):
    print('\n=== V5 MAE Pretraining ===')
    ds = VoxelDataset(f'{BASE}/morning/voxel')
    ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    model = V5MAEModel().to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    best = float('inf')

    for ep in range(1, epochs+1):
        model.train()
        total = 0
        for vox in tqdm(ld, desc=f'Ep{ep:02d}/{epochs}'):
            vox = vox.to(DEVICE)
            opt.zero_grad()
            loss = model(vox)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        sch.step()
        tl = total / len(ld)
        print(f'Ep {ep:02d} | mae_loss={tl:.4f}')

        if tl < best:
            best = tl
            # save tokenizer state for V5 fusion
            torch.save(model.tok.state_dict(),
                       CKPT_DIR/'v5_tokenizer_pretrained.pt')
            torch.save(model.state_dict(),
                       CKPT_DIR/'v5_mae_pretrained.pt')
            print(f'  Saved! loss={tl:.4f}')

    print(f'\nDone! Best mae_loss={best:.4f}')
    print('Next: use v5_tokenizer_pretrained.pt in fusion training')


if __name__ == '__main__':
    train(epochs=20)
