import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
from torch.utils.data import Dataset, DataLoader

DEVICE   = 'cuda'
BASE_OLD = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'
BASE_DVS = '/home/ivlab/carla_ws/examples'
CKPT_DIR = Path('/home/ivlab/carla_ws/fusion_model/checkpoints')
IMG_SIZE = 640
NC       = 4
H, W     = 720, 1280


def events_to_membrane_fast(events, H=H, W=W, decay=0.9, n_bins=10):
    V_pos = np.zeros((H, W), dtype=np.float32)
    V_neg = np.zeros((H, W), dtype=np.float32)
    if len(events) == 0:
        return np.stack([V_pos, V_neg])
    idx  = np.argsort(events['t'])
    evts = events[idx]
    xs   = np.clip(evts['x'].astype(np.int32), 0, W-1)
    ys   = np.clip(evts['y'].astype(np.int32), 0, H-1)
    pols = evts['pol'].astype(np.bool_)
    bins = np.array_split(np.arange(len(evts)), n_bins)
    for bin_idx in bins:
        if len(bin_idx) == 0: continue
        V_pos *= decay
        V_neg *= decay
        bx = xs[bin_idx]; by = ys[bin_idx]; bp = pols[bin_idx]
        np.add.at(V_pos, (by[bp],  bx[bp]),  1.0)
        np.add.at(V_neg, (by[~bp], bx[~bp]), 1.0)
    p_max = V_pos.max(); n_max = V_neg.max()
    if p_max > 0: V_pos /= p_max
    if n_max > 0: V_neg /= n_max
    return np.stack([V_pos, V_neg])


class MembranePotentialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.SiLU())
        self.p4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.SiLU())

    def forward(self, x):
        f  = self.stem(x)
        return self.p3(f), self.p4(f)


class MembraneFusion(nn.Module):
    def __init__(self, rgb_ch, mem_ch, n_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(mem_ch, rgb_ch, 1)
        self.ca   = nn.MultiheadAttention(rgb_ch, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(rgb_ch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(rgb_ch, 1), nn.Sigmoid())

    def forward(self, rgb_f, mem_f):
        B,C,H,W = rgb_f.shape
        mem = F.interpolate(self.proj(mem_f),(H,W),mode='bilinear',align_corners=False)
        gate = (1.0 - self.gate(rgb_f)).view(B,1,1)
        q = rgb_f.flatten(2).transpose(1,2)
        k = v = mem.flatten(2).transpose(1,2)
        o,_ = self.ca(q,k,v)
        return self.norm(q + o*gate).transpose(1,2).reshape(B,C,H,W)


class MembraneV5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_enc  = MembranePotentialEncoder()
        self.fuse_p3  = MembraneFusion(64,  64,  4)
        self.fuse_p4  = MembraneFusion(128, 128, 4)

        from ultralytics import YOLO
        m = YOLO('yolov8n.pt').model.model
        self.l0=m[0];self.l1=m[1];self.l2=m[2];self.l3=m[3]
        self.l4=m[4];self.l5=m[5];self.l6=m[6];self.l7=m[7]
        self.l8=m[8];self.l9=m[9];self.l10=m[10];self.l11=m[11]
        self.l12=m[12];self.l13=m[13];self.l14=m[14];self.l15=m[15]
        self.l16=m[16];self.l17=m[17];self.l18=m[18];self.l19=m[19]
        self.l20=m[20];self.l21=m[21];self.l22=m[22]

    def forward(self, rgb, membrane):
        # membrane encoder
        mp3, mp4 = self.mem_enc(membrane)  # (B,64,90,160) (B,128,45,80)

        # YOLOv8 backbone
        x0=self.l0(rgb);x1=self.l1(x0);x2=self.l2(x1)
        x3=self.l3(x2);x4=self.l4(x3);x5=self.l5(x4)
        x6=self.l6(x5)

        # Feature-level membrane fusion
        x4f = self.fuse_p3(x4, mp3)  # RGB P3 + membrane P3
        x6f = self.fuse_p4(x6, mp4)  # RGB P4 + membrane P4

        # YOLOv8 neck + head
        x7=self.l7(x6);x8=self.l8(x7);x9=self.l9(x8)
        x10=self.l10(x9)
        x11=self.l11([x10,x6f])
        x12=self.l12(x11);x13=self.l13(x12)
        x14=self.l14([x13,x4f])
        x15=self.l15(x14);x16=self.l16(x15)
        x17=self.l17([x16,x12])
        x18=self.l18(x17);x19=self.l19(x18)
        x20=self.l20([x19,x9])
        x21=self.l21(x20)
        return self.l22([x15,x18,x21])


# ── Generate membrane fusion images for ultralytics ──
def gen_membrane_overlay(split, rgb_dir, dvs_dir, label_dir,
                          out_img, out_lbl):
    """
    Generate membrane potential overlay images.
    Green = V_pos (brightness increase)
    Red   = V_neg (brightness decrease)
    This is NOT just visual — it's the physical signal.
    """
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)

    files = sorted(Path(rgb_dir).glob('frame_*.png'))
    done  = set(f.stem for f in Path(out_img).glob('*.png'))
    todo  = [f for f in files if f.stem not in done]
    print(f'{split}: {len(todo)} remaining')

    for rgb_path in tqdm(todo, desc=split):
        stem = rgb_path.stem
        dp   = Path(dvs_dir)/f'{stem}.npy'
        lp   = Path(label_dir)/f'{stem}.txt'
        if not dp.exists() or not lp.exists():
            continue

        # RGB CLAHE
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.resize(rgb,(IMG_SIZE,IMG_SIZE))
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        rgb_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Membrane potential
        events = np.load(str(dp))
        mp = events_to_membrane_fast(events)  # (2,H,W)

        # Resize membrane to IMG_SIZE
        mp_pos = cv2.resize(mp[0],(IMG_SIZE,IMG_SIZE))
        mp_neg = cv2.resize(mp[1],(IMG_SIZE,IMG_SIZE))

        # Create membrane overlay
        mem_vis = np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.float32)
        mem_vis[:,:,1] = mp_pos * 255  # green = V_pos
        mem_vis[:,:,2] = mp_neg * 255  # red   = V_neg

        # Blend: CLAHE RGB + membrane signal
        fused = cv2.addWeighted(
            rgb_enh.astype(np.float32), 0.65,
            mem_vis, 0.35, 0).clip(0,255).astype(np.uint8)

        cv2.imwrite(str(Path(out_img)/f'{stem}.png'), fused)
        shutil.copy(str(lp), str(Path(out_lbl)/f'{stem}.txt'))


if __name__ == '__main__':
    from ultralytics import YOLO

    out_base = '/home/ivlab/carla_ws/fusion_model/fusion_yolo_membrane'

    # Training data: morning_3_cameras (raw DVS available)
    gen_membrane_overlay('train',
        f'{BASE_DVS}/morning_3_cameras/rgb',
        f'{BASE_DVS}/morning_3_cameras/dvs',
        f'{BASE_OLD}/morning_3cam/yolo/labels',
        f'{out_base}/images/train',
        f'{out_base}/labels/train')

    # Val data: night_3_cameras
    gen_membrane_overlay('val',
        f'{BASE_DVS}/night_3_cameras/rgb',
        f'{BASE_DVS}/night_3_cameras/dvs',
        f'{BASE_OLD}/night_3cam/yolo/labels',
        f'{out_base}/images/val',
        f'{out_base}/labels/val')

    with open(f'{out_base}/data.yaml','w') as f:
        f.write(f"""path: {out_base}
train: images/train
val:   images/val
nc: 4
names: ['vehicle','pedestrian','traffic_sign','traffic_light']
""")

    model = YOLO('yolov8n.pt')
    model.train(
        data    = f'{out_base}/data.yaml',
        epochs  = 50, imgsz=IMG_SIZE,
        batch   = 8,  device=0,
        project = '/home/ivlab/carla_ws/fusion_model/runs',
        name    = 'fusion_membrane_v5',
        exist_ok= True, workers=0,
    )
    print('Membrane V5 training done!')
