import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

SRC  = '/home/ivlab/carla_ws/examples/morning_new'
DST  = '/home/ivlab/carla_ws/fusion_model/morning_new'
B    = 5
H, W = 720, 1280

def events_to_voxel_grid(events, B, H, W):
    voxel = np.zeros((B, H, W), dtype=np.float16)
    if len(events) == 0:
        return voxel
    t_min   = events['t'].min()
    t_max   = events['t'].max()
    t_range = t_max - t_min
    if t_range == 0:
        return voxel
    t_norm = (events['t'] - t_min) / t_range
    bins   = np.clip((t_norm * B).astype(int), 0, B-1)
    pol    = events['pol'].astype(np.float16)*2 - 1
    xs     = np.clip(events['x'], 0, W-1)
    ys     = np.clip(events['y'], 0, H-1)
    for i in range(len(events)):
        voxel[bins[i], ys[i], xs[i]] += pol[i]
    # normalize
    m = np.abs(voxel).max()
    if m > 0:
        voxel /= m
    return voxel

# create output dirs
for d in ['rgb', 'voxel', 'yolo/images', 'yolo/labels']:
    Path(f'{DST}/{d}').mkdir(parents=True, exist_ok=True)

# Step 1: copy RGB
print('Step 1: Copying RGB...')
rgb_files = sorted(Path(f'{SRC}/rgb').glob('frame_*.png'))
print(f'  {len(rgb_files)} frames')
for f in tqdm(rgb_files):
    shutil.copy(str(f), f'{DST}/rgb/{f.name}')
    shutil.copy(str(f), f'{DST}/yolo/images/{f.name}')

# Step 2: dvs → voxel
print('Step 2: Converting DVS to voxel...')
dvs_files = sorted(Path(f'{SRC}/dvs').glob('frame_*.npy'))
for f in tqdm(dvs_files):
    events = np.load(str(f))
    voxel  = events_to_voxel_grid(events, B, H, W)
    np.save(f'{DST}/voxel/{f.name}', voxel)

# Step 3: auto-label with RGB-only model
print('Step 3: Auto-labeling with RGB-only model...')
from ultralytics import YOLO
model = YOLO('/home/ivlab/carla_ws/fusion_model/runs/detect/runs/rgb_only_yolov8/weights/best.pt')

label_dir = Path(f'{DST}/yolo/labels')
for rgb_path in tqdm(rgb_files):
    results = model.predict(
        str(rgb_path), conf=0.3, verbose=False, imgsz=640)
    stem = rgb_path.stem
    with open(label_dir/f'{stem}.txt', 'w') as f:
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                xywh = box.xywh[0].cpu().numpy()
                x,y,w,h = xywh/640.0  # normalize
                f.write(f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

# Step 4: write data.yaml
with open(f'{DST}/data.yaml', 'w') as f:
    f.write(f"""path: {DST}
train: yolo/images
val: /home/ivlab/carla_ws/fusion_model/night/yolo/images
nc: 4
names: ['vehicle','pedestrian','traffic_sign','traffic_light']
""")

print(f'\nDone! Data saved to {DST}')
print('Next: train V3 with this new data')
