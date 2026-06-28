import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

DEVICE   = 'cuda'
BASE     = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'
CKPT_DIR = Path('/home/ivlab/carla_ws/fusion_model/checkpoints')
IMG_SIZE = 640
NC       = 4
PATCH    = 16
N_TOKENS = (IMG_SIZE // PATCH) ** 2
KEEP_K   = int(N_TOKENS * 0.4)


# ── Polarity-Aware Tokenizer ──
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
        pos_max = pos.flatten(2).max(-1)[0][...,None,None] + 1e-6
        neg_max = neg.flatten(2).max(-1)[0][...,None,None] + 1e-6
        pos = pos / pos_max
        neg = neg / neg_max
        pos_d = F.avg_pool2d(pos.mean(1,keepdim=True),PATCH,PATCH).flatten(1)
        neg_d = F.avg_pool2d(neg.mean(1,keepdim=True),PATCH,PATCH).flatten(1)
        density = torch.cat([pos_d, neg_d], dim=1)
        pos_tok = self.pos_embed(pos).flatten(2).transpose(1,2)
        neg_tok = self.neg_embed(neg).flatten(2).transpose(1,2)
        dev = voxel.device
        pos_tok += self.polarity_embed(torch.zeros(1,dtype=torch.long,device=dev))
        neg_tok += self.polarity_embed(torch.ones(1, dtype=torch.long,device=dev))
        return self.norm_pos(pos_tok), self.norm_neg(neg_tok), density


class SparseSelector(nn.Module):
    def __init__(self, keep_k=KEEP_K):
        super().__init__()
        self.k = keep_k

    def forward(self, tokens, density):
        _, idx = density.topk(self.k, dim=1)
        return tokens.gather(1, idx.unsqueeze(-1).expand(-1,-1,tokens.shape[-1]))


class ViTBlock(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x):
        h,_ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x   = x + h
        return x + self.mlp(self.norm2(x))


class V5EventEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=6):
        super().__init__()
        self.tok  = PolarityAwareTokenizer(5, embed_dim)
        self.sel  = SparseSelector(KEEP_K)
        self.pe   = nn.Parameter(torch.randn(1, KEEP_K, embed_dim)*0.02)
        self.blks = nn.ModuleList([ViTBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, voxel):
        pt, nt, density = self.tok(voxel)
        tokens = torch.cat([pt, nt], dim=1)
        tokens = self.sel(tokens, density) + self.pe
        for blk in self.blks:
            tokens = blk(tokens)
        return self.norm(tokens)


class IllumFusion(nn.Module):
    def __init__(self, rgb_ch, edim=256, heads=4):
        super().__init__()
        hd = rgb_ch // heads
        self.heads = heads; self.hd = hd; self.scale = hd**-0.5
        self.q = nn.Linear(rgb_ch, rgb_ch)
        self.k = nn.Linear(edim, rgb_ch)
        self.v = nn.Linear(edim, rgb_ch)
        self.o = nn.Linear(rgb_ch, rgb_ch)
        self.n = nn.LayerNorm(rgb_ch)
        self.g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(rgb_ch,1), nn.Sigmoid())

    def forward(self, f, e):
        B,C,H,W = f.shape; K=e.shape[1]; nh=self.heads; hd=self.hd
        x = f.flatten(2).transpose(1,2)
        gate = (1 - self.g(f)).unsqueeze(1)
        Q = self.q(x).view(B,H*W,nh,hd).transpose(1,2)
        K_= self.k(e).view(B,K,nh,hd).transpose(1,2)
        V_= self.v(e).view(B,K,nh,hd).transpose(1,2)
        a = (Q@K_.transpose(-2,-1))*self.scale
        o = (a.softmax(-1)@V_).transpose(1,2).reshape(B,H*W,C)
        return self.n(x + self.o(o)*gate).transpose(1,2).reshape(B,C,H,W)


# ── Generate V5 fusion images ──
def gen_fusion_images(split, rgb_dir, voxel_dir, label_dir,
                      out_img, out_lbl, encoder, device):
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)

    files = sorted(Path(rgb_dir).glob('frame_*.png'))
    done  = set(f.stem for f in Path(out_img).glob('*.png'))
    todo  = [f for f in files if f.stem not in done]
    print(f'{split}: {len(todo)} remaining')

    import shutil
    encoder.eval()
    with torch.no_grad():
        for rgb_path in tqdm(todo, desc=split):
            stem = rgb_path.stem
            vp   = Path(voxel_dir)/f'{stem}.npy'
            lp   = Path(label_dir)/f'{stem}.txt'
            if not vp.exists() or not lp.exists():
                continue

            # RGB + CLAHE
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            rgb_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Voxel → V5 event tokens
            v    = np.load(str(vp)).astype(np.float32)
            bins = [cv2.resize(v[b],(IMG_SIZE,IMG_SIZE)) for b in range(5)]
            vt   = torch.from_numpy(np.stack(bins)).unsqueeze(0).to(device)

            # Get polarity-aware tokens
            pt, nt, density = encoder.tok(vt)
            # Use positive tokens for green, negative for red
            pos_attn = pt[0].norm(dim=-1).reshape(40,40).cpu().numpy()
            neg_attn = nt[0].norm(dim=-1).reshape(40,40).cpu().numpy()
            pos_attn = cv2.resize(pos_attn,(IMG_SIZE,IMG_SIZE))
            neg_attn = cv2.resize(neg_attn,(IMG_SIZE,IMG_SIZE))
            pos_attn = (pos_attn-pos_attn.min())/(pos_attn.max()-pos_attn.min()+1e-6)
            neg_attn = (neg_attn-neg_attn.min())/(neg_attn.max()-neg_attn.min()+1e-6)

            # Raw polarity
            vsum = v.sum(0)
            vsum = cv2.resize(vsum,(IMG_SIZE,IMG_SIZE))
            pos_pol = np.clip( vsum*5,0,1)*255
            neg_pol = np.clip(-vsum*5,0,1)*255

            # Polarity-aware weighted overlay
            evt = np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.float32)
            evt[:,:,1] = pos_pol * pos_attn  # green = positive
            evt[:,:,2] = neg_pol * neg_attn  # red   = negative

            fused = cv2.addWeighted(
                rgb_enh.astype(np.float32), 0.65,
                evt, 0.35, 0).clip(0,255).astype(np.uint8)

            cv2.imwrite(str(Path(out_img)/f'{stem}.png'), fused)
            shutil.copy(str(lp), str(Path(out_lbl)/f'{stem}.txt'))


if __name__ == '__main__':
    from ultralytics import YOLO

    # Build V5 encoder
    encoder = V5EventEncoder(embed_dim=256, depth=6).to(DEVICE)
    # Load pretrained polarity-aware tokenizer
    from pathlib import Path
    ckpt = Path('/home/ivlab/carla_ws/fusion_model/checkpoints/v5_tokenizer_pretrained.pt')
    if ckpt.exists():
        encoder.tok.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        print('Loaded pretrained V5 tokenizer!')
    else:
        print('No pretrained tokenizer found, using random init')

    out_base = '/home/ivlab/carla_ws/fusion_model/fusion_yolo_v5'

    gen_fusion_images('train',
        f'{BASE}/morning/rgb', f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels',
        f'{out_base}/images/train',
        f'{out_base}/labels/train',
        encoder, DEVICE)

    gen_fusion_images('val',
        f'{BASE}/night/rgb', f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels',
        f'{out_base}/images/val',
        f'{out_base}/labels/val',
        encoder, DEVICE)

    # Write yaml
    with open(f'{out_base}/data.yaml','w') as f:
        f.write(f"""path: {out_base}
train: images/train
val:   images/val
nc: 4
names: ['vehicle','pedestrian','traffic_sign','traffic_light']
""")

    # Train with ultralytics
    model = YOLO('yolov8n.pt')
    model.train(
        data    = f'{out_base}/data.yaml',
        epochs  = 50,
        imgsz   = IMG_SIZE,
        batch   = 8,
        device  = 0,
        project = '/home/ivlab/carla_ws/fusion_model/runs',
        name    = 'fusion_yolov8_v5',
        exist_ok= True,
        workers = 0,
    )
    print('V5 training done!')
