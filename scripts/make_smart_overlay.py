import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from model.rsgnet import RSGNet

DEVICE   = 'cuda'
BASE     = '/home/ivlab/carla_ws/fusion_model'
IMG_SIZE = 640

rsgnet = RSGNet(in_ch=5, out_ch=3).to(DEVICE)
ckpt   = torch.load(f'{BASE}/checkpoints/best.pt', map_location=DEVICE)
rsg_w  = {k.replace('rsgnet.',''):v for k,v in ckpt.items()
          if k.startswith('rsgnet.')}
rsgnet.load_state_dict(rsg_w)
rsgnet.eval()
print('RSGNet loaded!')

def smart_overlay(rgb_bgr, sif_np):
    """
    智慧疊加：只在暗區加強 SIF
    亮區保留 RGB，暗區用 SIF 補強
    """
    # 計算每個像素的亮度
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    brightness = gray.astype(np.float32) / 255.0

    # 暗區權重高，亮區權重低
    dark_mask = (1.0 - brightness)               # (H,W)
    dark_mask = cv2.GaussianBlur(dark_mask, (21,21), 0)
    dark_mask = np.clip(dark_mask * 1.5, 0, 0.6) # max 60% SIF

    # SIF colormap
    sif_vis   = (sif_np.mean(0) * 255).astype(np.uint8)
    sif_color = cv2.applyColorMap(sif_vis, cv2.COLORMAP_INFERNO)

    # pixel-wise blend
    dark_mask_3ch = np.stack([dark_mask]*3, axis=-1)
    fused = (rgb_bgr.astype(np.float32) * (1 - dark_mask_3ch) +
             sif_color.astype(np.float32) * dark_mask_3ch)
    return fused.astype(np.uint8)


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
            if not voxel_path.exists() or not label_path.exists():
                continue

            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

            v    = np.load(str(voxel_path)).astype(np.float32)
            bins = [cv2.resize(v[b],(IMG_SIZE,IMG_SIZE))
                    for b in range(v.shape[0])]
            vt   = torch.from_numpy(
                np.stack(bins)).unsqueeze(0).to(DEVICE)
            sif  = rsgnet(vt)[0].cpu().numpy()

            fused = smart_overlay(rgb, sif)
            cv2.imwrite(str(Path(out_img)/f'{stem}.png'), fused)
            shutil.copy(str(label_path),
                        str(Path(out_lbl)/f'{stem}.txt'))

process('train',
        f'{BASE}/morning/rgb', f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels',
        f'{BASE}/fusion_yolo_v2/images/train',
        f'{BASE}/fusion_yolo_v2/labels/train')

process('val',
        f'{BASE}/night/rgb', f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels',
        f'{BASE}/fusion_yolo_v2/images/val',
        f'{BASE}/fusion_yolo_v2/labels/val')

print('Done!')
