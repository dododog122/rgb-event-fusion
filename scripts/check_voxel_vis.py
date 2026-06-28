import cv2
import numpy as np
from pathlib import Path

BASE = '/home/ivlab/carla_ws/fusion_model'

for i in range(5):
    stem = f'frame_{i:06d}'
    rgb  = cv2.imread(f'{BASE}/night/rgb/{stem}.png')
    rgb  = cv2.resize(rgb, (640,640))
    v    = np.load(f'{BASE}/night/voxel/{stem}.npy').astype(np.float32)

    # sum all bins
    v_sum = v.sum(0)  # (720,1280)
    v_sum = cv2.resize(v_sum, (640,640))

    pos = np.clip( v_sum, 0, None)
    neg = np.clip(-v_sum, 0, None)

    # amplify to make visible
    pos = np.clip(pos * 5, 0, 1)
    neg = np.clip(neg * 5, 0, 1)

    evt = np.zeros((640,640,3), dtype=np.float32)
    evt[:,:,1] = pos * 255   # green
    evt[:,:,2] = neg * 255   # red

    # CLAHE on RGB
    lab   = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    rgb_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    fused = cv2.addWeighted(
        rgb_enh.astype(np.float32), 0.65,
        evt, 0.35, 0).clip(0,255).astype(np.uint8)

    # side by side: RGB | event only | fused
    evt_vis = evt.clip(0,255).astype(np.uint8)
    combined = np.hstack([rgb, evt_vis, fused])
    cv2.imwrite(f'/tmp/voxel_vis_{i}.png', combined)
    print(f'{stem}: nonzero={( v_sum != 0).sum()} fused_mean={fused.mean():.1f}')

print('Saved to /tmp/voxel_vis_*.png')
