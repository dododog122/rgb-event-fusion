"""
把 RGB (3ch) 和 SIF (3ch) 合併存成兩張圖：
- frame_XXXXXX_rgb.png  ← 正常 RGB
- frame_XXXXXX_sif.png  ← SIF 視覺化 (用於 demo)
並且把 SIF 疊加到 RGB 上存成 fusion PNG 給 YOLOv8 訓練
"""
import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import shutil
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model.rsgnet import RSGNet

DEVICE   = 'cuda'
BASE     = '/home/ivlab/carla_ws/fusion_model'
IMG_SIZE = 640
ALPHA    = 0.4   # SIF overlay strength

rsgnet = RSGNet(in_ch=5, out_ch=3).to(DEVICE)
ckpt   = torch.load(f'{BASE}/checkpoints/best.pt',
                    map_location=DEVICE)
rsg_w  = {k.replace('rsgnet.',''):v
          for k,v in ckpt.items()
          if k.startswith('rsgnet.')}
rsgnet.load_state_dict(rsg_w)
rsgnet.eval()
print('RSGNet loaded!')

def process(split, rgb_dir, voxel_dir, label_dir,
            out_img, out_lbl):
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)

    files = sorted(Path(rgb_dir).glob('frame_*.png'))
    print(f'{split}: {len(files)} frames')

    with torch.no_grad():
        for rgb_path in tqdm(files, desc=split):
            stem       = rgb_path.stem
            voxel_path = Path(voxel_dir)/f'{stem}.npy'
            label_path = Path(label_dir)/f'{stem}.txt'
            if not voxel_path.exists() or \
               not label_path.exists():
                continue

            # RGB
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

            # Voxel → SIF
            v = np.load(str(voxel_path)).astype(np.float32)
            bins = [cv2.resize(v[b], (IMG_SIZE, IMG_SIZE))
                    for b in range(v.shape[0])]
            vt = torch.from_numpy(
                np.stack(bins)).unsqueeze(0).to(DEVICE)
            sif = rsgnet(vt)[0].cpu().numpy()  # (3,H,W)

            # SIF → colormap overlay
            sif_vis = (sif.mean(0) * 255).astype(np.uint8)
            sif_color = cv2.applyColorMap(
                sif_vis, cv2.COLORMAP_INFERNO)

            # blend RGB + SIF
            fused = cv2.addWeighted(
                rgb, 1-ALPHA, sif_color, ALPHA, 0)

            # save fused image
            cv2.imwrite(
                str(Path(out_img)/f'{stem}.png'), fused)

            # copy label
            shutil.copy(str(label_path),
                        str(Path(out_lbl)/f'{stem}.txt'))

process('train',
        f'{BASE}/morning/rgb',
        f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels',
        f'{BASE}/fusion_yolo/images/train',
        f'{BASE}/fusion_yolo/labels/train')

process('val',
        f'{BASE}/night/rgb',
        f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels',
        f'{BASE}/fusion_yolo/images/val',
        f'{BASE}/fusion_yolo/labels/val')

print('\nDone!')
