"""
V5: Polarity-Aware Dual Embedding + Sparse Token Selection
- Softplus polarity separation (not ReLU)
- Learnable temperature for each polarity
- Per-bin normalization
- Polarity type embedding
- Sparse token selection (top 40% by event density)
- Cross-attention fusion with YOLOv8 P3/P4
"""

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

DEVICE   = 'cuda'
BASE     = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'
CKPT_DIR = Path('/home/ivlab/carla_ws/fusion_model/checkpoints')
IMG_SIZE = 640
NC       = 4
PATCH    = 16
N_TOKENS = (IMG_SIZE // PATCH) ** 2          # 1600
KEEP_K   = int(N_TOKENS * 0.4)              # 640 sparse tokens


# ══════════════════════════════════════════
# Polarity-Aware Tokenizer
# ══════════════════════════════════════════
class PolarityAwareTokenizer(nn.Module):
    """
    Split voxel into positive/negative branches.
    Use Softplus (not ReLU) for smooth separation.
    Learnable temperature controls filter strength.
    """
    def __init__(self, in_ch=5, embed_dim=256):
        super().__init__()

        # Learnable temperature (Softplus gate)
        self.pos_temp = nn.Parameter(torch.ones(1) * 2.0)
        self.neg_temp = nn.Parameter(torch.ones(1) * 2.0)

        # Separate patch embeddings for pos and neg
        self.pos_embed = nn.Conv2d(in_ch, embed_dim,
                                    kernel_size=PATCH, stride=PATCH, bias=False)
        self.neg_embed = nn.Conv2d(in_ch, embed_dim,
                                    kernel_size=PATCH, stride=PATCH, bias=False)

        # Polarity type embedding: 0=positive, 1=negative
        self.polarity_embed = nn.Embedding(2, embed_dim)

        # Layer norm after tokenization
        self.norm_pos = nn.LayerNorm(embed_dim)
        self.norm_neg = nn.LayerNorm(embed_dim)

    def forward(self, voxel):
        """
        voxel: (B, 5, H, W)  values in [-1, +1]
        returns: tokens (B, 2*N_TOKENS, embed_dim)
                 density (B, 2*N_TOKENS) for sparse selection
        """
        B = voxel.shape[0]

        # ── Soft polarity separation ──
        # Softplus: smooth, always has gradient, learnable threshold
        pos = F.softplus(voxel * self.pos_temp)    # brightness increase
        neg = F.softplus(-voxel * self.neg_temp)   # brightness decrease

        # ── Per-bin normalization ──
        # Each temporal bin normalized independently
        pos_max = pos.flatten(2).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-6
        neg_max = neg.flatten(2).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-6
        pos = pos / pos_max   # (B, 5, H, W) normalized
        neg = neg / neg_max   # (B, 5, H, W) normalized

        # ── Compute event density per patch (for sparse selection) ──
        # density = mean absolute value in each patch
        pos_density = F.avg_pool2d(pos.mean(1, keepdim=True),
                                    PATCH, PATCH).flatten(1)  # (B, N)
        neg_density = F.avg_pool2d(neg.mean(1, keepdim=True),
                                    PATCH, PATCH).flatten(1)  # (B, N)
        density = torch.cat([pos_density, neg_density], dim=1)  # (B, 2N)

        # ── Separate patch embedding ──
        pos_tok = self.pos_embed(pos).flatten(2).transpose(1, 2)   # (B, N, D)
        neg_tok = self.neg_embed(neg).flatten(2).transpose(1, 2)   # (B, N, D)

        # ── Add polarity type embedding ──
        pol_idx_pos = torch.zeros(1, dtype=torch.long, device=voxel.device)
        pol_idx_neg = torch.ones(1,  dtype=torch.long, device=voxel.device)
        pos_tok = pos_tok + self.polarity_embed(pol_idx_pos)  # "I am positive"
        neg_tok = neg_tok + self.polarity_embed(pol_idx_neg)  # "I am negative"

        # ── Normalize ──
        pos_tok = self.norm_pos(pos_tok)  # (B, N, D)
        neg_tok = self.norm_neg(neg_tok)  # (B, N, D)

        tokens = torch.cat([pos_tok, neg_tok], dim=1)  # (B, 2N, D)
        return tokens, density


# ══════════════════════════════════════════
# Sparse Token Selection
# ══════════════════════════════════════════
class SparseTokenSelector(nn.Module):
    """
    Keep only top-K tokens by event density.
    Reduces computation by 60% while focusing on active regions.
    """
    def __init__(self, keep_k=KEEP_K):
        super().__init__()
        self.keep_k = keep_k

    def forward(self, tokens, density):
        """
        tokens:  (B, 2N, D)
        density: (B, 2N)
        returns: sparse_tokens (B, K, D)
        """
        B = tokens.shape[0]
        _, topk_idx = density.topk(self.keep_k, dim=1)  # (B, K)
        # gather selected tokens
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        sparse = tokens.gather(1, idx)  # (B, K, D)
        return sparse


# ══════════════════════════════════════════
# ViT Encoder Block
# ══════════════════════════════════════════
class ViTBlock(nn.Module):
    def __init__(self, dim=256, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ══════════════════════════════════════════
# V5 Event Encoder
# ══════════════════════════════════════════
class V5EventEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=6, n_heads=8):
        super().__init__()
        self.tokenizer = PolarityAwareTokenizer(in_ch=5, embed_dim=embed_dim)
        self.selector  = SparseTokenSelector(keep_k=KEEP_K)
        self.pos_enc   = nn.Parameter(torch.randn(1, KEEP_K, embed_dim) * 0.02)
        self.blocks    = nn.ModuleList([
            ViTBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, voxel):
        """
        voxel: (B, 5, H, W)
        returns: event_tokens (B, K, D)  K=640 sparse tokens
        """
        tokens, density = self.tokenizer(voxel)      # (B, 2N, D)
        tokens = self.selector(tokens, density)       # (B, K, D)
        tokens = tokens + self.pos_enc                # add position encoding
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)                      # (B, K, D)


# ══════════════════════════════════════════
# Cross-Attention Fusion (Illumination-Aware)
# ══════════════════════════════════════════
class IlluminationAwareFusion(nn.Module):
    """
    RGB feature attends to event tokens.
    Illumination gate: dark scene → rely more on event.
    """
    def __init__(self, rgb_ch, event_dim=256, n_heads=4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = rgb_ch // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q    = nn.Linear(rgb_ch,    rgb_ch)
        self.k    = nn.Linear(event_dim, rgb_ch)
        self.v    = nn.Linear(event_dim, rgb_ch)
        self.out  = nn.Linear(rgb_ch,    rgb_ch)
        self.norm = nn.LayerNorm(rgb_ch)

        # Illumination gate: darker → more event contribution
        self.illum_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(rgb_ch, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, event_tokens):
        B, C, H, W = rgb_feat.shape
        K = event_tokens.shape[1]
        nh = self.n_heads
        hd = self.head_dim

        # Illumination gate
        gate = self.illum_gate(rgb_feat)   # (B, 1) high=bright, low=dark
        event_weight = 1.0 - gate          # dark → weight up event

        # Flatten RGB spatial
        x = rgb_feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        Q = self.q(x).view(B, H*W, nh, hd).transpose(1, 2)
        K_ = self.k(event_tokens).view(B, K, nh, hd).transpose(1, 2)
        V_ = self.v(event_tokens).view(B, K, nh, hd).transpose(1, 2)

        attn = (Q @ K_.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ V_).transpose(1, 2).reshape(B, H*W, C)
        out  = self.out(out)

        # Apply illumination-aware weighting
        fused = self.norm(x + out * event_weight.unsqueeze(1))
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ══════════════════════════════════════════
# V5 Full Model
# ══════════════════════════════════════════
class V5Model(nn.Module):
    def __init__(self):
        super().__init__()

        # V5 Event Encoder
        self.event_enc = V5EventEncoder(embed_dim=256, depth=6, n_heads=8)

        # YOLOv8n backbone + neck + head
        yolo = YOLO('yolov8n.pt')
        m = yolo.model.model
        self.l0  = m[0];  self.l1  = m[1];  self.l2  = m[2]
        self.l3  = m[3];  self.l4  = m[4];  self.l5  = m[5]
        self.l6  = m[6];  self.l7  = m[7];  self.l8  = m[8]
        self.l9  = m[9];  self.l10 = m[10]; self.l11 = m[11]
        self.l12 = m[12]; self.l13 = m[13]; self.l14 = m[14]
        self.l15 = m[15]; self.l16 = m[16]; self.l17 = m[17]
        self.l18 = m[18]; self.l19 = m[19]; self.l20 = m[20]
        self.l21 = m[21]; self.l22 = m[22]

        # Illumination-aware fusion at P3(64ch) and P4(128ch)
        self.fuse_p3 = IlluminationAwareFusion(rgb_ch=64,  event_dim=256, n_heads=4)
        self.fuse_p4 = IlluminationAwareFusion(rgb_ch=128, event_dim=256, n_heads=4)

    def forward(self, rgb, voxel):
        # V5 event encoding
        evt = self.event_enc(voxel)   # (B, 640, 256) sparse tokens

        # YOLOv8 backbone with fusion
        x0  = self.l0(rgb)
        x1  = self.l1(x0)
        x2  = self.l2(x1)
        x3  = self.l3(x2)
        x4  = self.l4(x3)            # P3 (B,64,80,80)
        x5  = self.l5(x4)
        x6  = self.l6(x5)            # P4 (B,128,40,40)

        # Feature-level fusion
        x4f = self.fuse_p3(x4, evt)  # fused P3
        x6f = self.fuse_p4(x6, evt)  # fused P4

        x7  = self.l7(x6)
        x8  = self.l8(x7)
        x9  = self.l9(x8)
        x10 = self.l10(x9)
        x11 = self.l11([x10, x6f])   # Concat with fused P4
        x12 = self.l12(x11)
        x13 = self.l13(x12)
        x14 = self.l14([x13, x4f])   # Concat with fused P3
        x15 = self.l15(x14)
        x16 = self.l16(x15)
        x17 = self.l17([x16, x12])
        x18 = self.l18(x17)
        x19 = self.l19(x18)
        x20 = self.l20([x19, x9])
        x21 = self.l21(x20)
        out = self.l22([x15, x18, x21])
        return out


# ══════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════
class V5Dataset(Dataset):
    def __init__(self, rgb_dir, voxel_dir, label_dir):
        self.vd = Path(voxel_dir)
        self.ld = Path(label_dir)
        self.files = sorted([
            f for f in Path(rgb_dir).glob('frame_*.png')
            if (self.vd/f'{f.stem}.npy').exists()
            and (self.ld/f'{f.stem}.txt').exists()
        ])
        print(f'  {len(self.files)} frames')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx].stem
        S    = IMG_SIZE

        rgb = cv2.imread(str(self.files[idx]))
        rgb = cv2.resize(rgb, (S, S))
        rgb = torch.from_numpy(rgb).permute(2,0,1).float() / 255.

        v   = np.load(str(self.vd/f'{stem}.npy')).astype(np.float32)
        vox = torch.from_numpy(
            np.stack([cv2.resize(v[b],(S,S)) for b in range(5)]))

        boxes, cls_ids = [], []
        with open(self.ld/f'{stem}.txt') as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    c,x,y,w,h = map(float, p)
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
# Loss (same as V4 but cleaner)
# ══════════════════════════════════════════
def compute_loss(pred_out, gt_boxes_list, gt_cls_list, device='cuda'):
    # training mode → dict; eval mode → tuple[tensor, dict]
    if isinstance(pred_out, dict):
        pred_cls = pred_out['scores'].permute(0,2,1)  # (B,8400,80)
        pred_box = pred_out['boxes'].permute(0,2,1)   # (B,8400,64)
    elif isinstance(pred_out, (list,tuple)):
        p = pred_out[0]                                # (B,84,8400)
        pred_box = p[:,:4,:].permute(0,2,1)
        pred_cls = p[:,4:,:].permute(0,2,1)
    else:
        return torch.tensor(0., device=device, requires_grad=True)

    B    = pred_cls.shape[0]
    N    = pred_cls.shape[1]
    pred_cls_nc = pred_cls[:,:,:NC]   # (B,8400,NC)

    # anchor grid
    anchors = []
    for sz in [80,40,20]:
        gy,gx = torch.meshgrid(
            torch.arange(sz,dtype=torch.float32,device=device),
            torch.arange(sz,dtype=torch.float32,device=device),
            indexing='ij')
        anchors.append(torch.stack([(gx+0.5)/sz,(gy+0.5)/sz],dim=-1).reshape(-1,2))
    anchors = torch.cat(anchors)  # (8400,2)

    all_losses = []

    for b in range(B):
        gt_b = gt_boxes_list[b].to(device)
        cl_b = gt_cls_list[b].to(device)

        obj_tgt = torch.zeros(N, device=device)
        obj_l   = F.binary_cross_entropy_with_logits(
                      pred_cls_nc[b].max(-1).values, obj_tgt)

        if gt_b.shape[0] == 0:
            all_losses.append(obj_l)
            continue

        dist = torch.cdist(anchors, gt_b[:,:2])           # (8400,M)
        _, topk = dist.topk(10, dim=0, largest=False)     # (10,M)
        pos_idx = topk.reshape(-1).unique()

        obj_tgt[pos_idx] = 1.
        obj_l2 = F.binary_cross_entropy_with_logits(
                     pred_cls_nc[b].max(-1).values, obj_tgt)

        # box targets
        box_tgt = torch.zeros(len(pos_idx), 4,  device=device)
        cls_tgt = torch.zeros(len(pos_idx), NC, device=device)
        for n in range(gt_b.shape[0]):
            pidx_n = topk[:,n]
            mask   = torch.isin(pos_idx, pidx_n)
            box_tgt[mask] = gt_b[n]
            cls_tgt[mask, cl_b[n]] = 1.

        box_l = F.mse_loss(
            pred_box[:,:,:4][b][pos_idx].sigmoid()
            if pred_box.shape[-1] >= 4
            else pred_box[b][pos_idx].sigmoid(),
            box_tgt)
        cls_l = F.binary_cross_entropy_with_logits(
                    pred_cls_nc[b][pos_idx], cls_tgt)

        all_losses.append(obj_l2 + box_l + cls_l)

    return torch.stack(all_losses).mean()


# ══════════════════════════════════════════
# Training
# ══════════════════════════════════════════
def train(epochs=50):
    print('\n=== V5 Polarity-Aware Feature-level Fusion ===')
    print(f'Sparse tokens: {KEEP_K}/{N_TOKENS*2} (40% kept)')

    train_ds = V5Dataset(
        f'{BASE}/morning/rgb',
        f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels')
    val_ds = V5Dataset(
        f'{BASE}/night/rgb',
        f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels')

    train_ld = DataLoader(train_ds, batch_size=2, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

    model = V5Model().to(DEVICE)
    print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # Phase A: freeze event encoder, train fusion + YOLO
    for p in model.event_enc.parameters():
        p.requires_grad = False

    trainable = (list(model.fuse_p3.parameters()) +
                 list(model.fuse_p4.parameters()) +
                 [p for name,p in model.named_parameters()
                  if name.startswith('l') and p.requires_grad])

    opt = AdamW(trainable, lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    best = float('inf')

    for ep in range(1, epochs+1):
        # Phase B: unfreeze event encoder at epoch 10
        if ep == 10:
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
                vl  += loss.item()

        sch.step()
        tl = tr / len(train_ld)
        vl = vl / len(val_ld)
        print(f'Ep {ep:02d} | train={tl:.4f}  val={vl:.4f}')

        if vl < best:
            best = vl
            torch.save(model.state_dict(),
                       CKPT_DIR/'v5_best.pt')
            print(f'  Saved! val={vl:.4f}')

    print(f'\nDone! Best val={best:.4f}')
    print('Next: run mAP evaluation with v5_best.pt')


if __name__ == '__main__':
    train(epochs=50)
