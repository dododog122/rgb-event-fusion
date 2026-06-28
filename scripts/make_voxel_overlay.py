import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

BASE     = '/home/ivlab/carla_ws/fusion_model'
IMG_SIZE = 640

def voxel_overlay(rgb_bgr, voxel_npy):
    # Step 1: CLAHE 提升暗部亮度
    lab   = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    rgb_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 2: Event voxel → 正負極性顏色
    v   = voxel_npy.astype(np.float32)
    pos = np.clip( v, 0, None).sum(0)
    neg = np.clip(-v, 0, None).sum(0)
    pos = (pos / (pos.max() + 1e-6) * 255)
    neg = (neg / (neg.max() + 1e-6) * 255)

    evt_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    evt_vis[:,:,1] = pos   # green = positive events
    evt_vis[:,:,2] = neg   # red   = negative events

    # Step 3: blend
    fused = cv2.addWeighted(
        rgb_enhanced.astype(np.float32), 0.7,
        evt_vis, 0.3, 0)
    return fused.clip(0,255).astype(np.uint8)


def process(split, rgb_dir, voxel_dir, label_dir,
            out_img, out_lbl):
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(rgb_dir).glob('frame_*.png'))
    print(f'{split}: {len(files)} frames')

    for rgb_path in tqdm(files, desc=split):
        stem       = rgb_path.stem
        voxel_path = Path(voxel_dir)/f'{stem}.npy'
        label_path = Path(label_dir)/f'{stem}.txt'
        if not voxel_path.exists() or not label_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        v    = np.load(str(voxel_path)).astype(np.float32)
        bins = [cv2.resize(v[b], (IMG_SIZE, IMG_SIZE))
                for b in range(v.shape[0])]
        v_rs = np.stack(bins)

        fused = voxel_overlay(rgb, v_rs)
        cv2.imwrite(str(Path(out_img)/f'{stem}.png'), fused)
        shutil.copy(str(label_path),
                    str(Path(out_lbl)/f'{stem}.txt'))


process('train',
        f'{BASE}/morning/rgb', f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels',
        f'{BASE}/fusion_yolo_v3/images/train',
        f'{BASE}/fusion_yolo_v3/labels/train')

process('val',
        f'{BASE}/night/rgb', f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels',
        f'{BASE}/fusion_yolo_v3/images/val',
        f'{BASE}/fusion_yolo_v3/labels/val')
print('Done!')
