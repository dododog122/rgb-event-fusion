import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

SPLITS = {
    'morning': '/home/ivlab/carla_ws/examples/morning_3_cameras',
    'night':   '/home/ivlab/carla_ws/examples/night_3_cameras',
}
DST_BASE = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'

SEM_TO_YOLO = {
    10: 0, 4: 1, 24: 1, 12: 2, 18: 3, 25: 0,
}

B = 5
H, W = 720, 1280
MIN_PIXELS = 50


def events_to_voxel(events, B, H, W):
    """Vectorized version — much faster."""
    voxel = np.zeros((B, H, W), dtype=np.float16)
    if len(events) == 0:
        return voxel
    t_min = events['t'].min()
    t_max = events['t'].max()
    t_range = t_max - t_min
    if t_range == 0:
        return voxel

    t_norm = (events['t'] - t_min) / t_range
    bins   = np.clip((t_norm * B).astype(np.int32), 0, B-1)
    pol    = events['pol'].astype(np.float32)*2 - 1
    xs     = np.clip(events['x'].astype(np.int32), 0, W-1)
    ys     = np.clip(events['y'].astype(np.int32), 0, H-1)

    # vectorized: use np.add.at
    np.add.at(voxel, (bins, ys, xs), pol.astype(np.float16))

    m = np.abs(voxel).max()
    if m > 0:
        voxel /= m
    return voxel


def semantic_to_yolo(sem_img):
    labels = []
    h, w   = sem_img.shape[:2]
    cls_map = sem_img[:,:,0]

    for sem_id, yolo_cls in SEM_TO_YOLO.items():
        mask = (cls_map == sem_id)
        if mask.sum() < MIN_PIXELS:
            continue
        mask_u8 = mask.astype(np.uint8) * 255
        n, comps = cv2.connectedComponents(mask_u8)
        for comp_id in range(1, n):
            comp_mask = (comps == comp_id)
            if comp_mask.sum() < MIN_PIXELS:
                continue
            ys, xs = np.where(comp_mask)
            x1,x2 = xs.min(),xs.max()
            y1,y2 = ys.min(),ys.max()
            cx = (x1+x2)/2/w; cy = (y1+y2)/2/h
            bw = (x2-x1)/w;   bh = (y2-y1)/h
            if bw > 0.005 and bh > 0.005:
                labels.append(f'{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return labels


def process_split(split_name, src_dir):
    print(f'\n=== {split_name} ===')
    src = Path(src_dir)
    dst = Path(f'{DST_BASE}/{split_name}_3cam')
    for d in ['rgb','voxel','yolo/images','yolo/labels']:
        (dst/d).mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(src.glob('rgb/frame_*.png'))
    done_vox  = set(f.stem for f in (dst/'voxel').glob('*.npy'))
    done_lbl  = set(f.stem for f in (dst/'yolo'/'labels').glob('*.txt'))
    todo = [f for f in rgb_files if f.stem not in done_vox]
    print(f'  Total={len(rgb_files)}  Done={len(done_vox)}  Remaining={len(todo)}')

    for rgb_path in tqdm(todo):
        stem = rgb_path.stem

        # RGB
        dst_img = dst/'yolo'/'images'/rgb_path.name
        if not dst_img.exists():
            shutil.copy(str(rgb_path), str(dst/'rgb'/rgb_path.name))
            shutil.copy(str(rgb_path), str(dst_img))

        # voxel
        dvs_path = src/f'dvs/{stem}.npy'
        if dvs_path.exists():
            events = np.load(str(dvs_path))
            voxel  = events_to_voxel(events, B, H, W)
            np.save(str(dst/f'voxel/{stem}.npy'), voxel)

        # labels
        if stem not in done_lbl:
            sem_path = src/f'semantic/{stem}.png'
            if sem_path.exists():
                sem_img = cv2.imread(str(sem_path))
                labels  = semantic_to_yolo(sem_img)
                with open(str(dst/'yolo'/'labels'/f'{stem}.txt'), 'w') as f:
                    f.write('\n'.join(labels))

    print(f'  Saved to {dst}')

for split, src in SPLITS.items():
    process_split(split, src)

with open(f'{DST_BASE}/data_3cam.yaml', 'w') as f:
    f.write(f"""path: {DST_BASE}
train: morning_3cam/yolo/images
val:   night_3cam/yolo/images
nc: 4
names: ['vehicle','pedestrian','traffic_sign','traffic_light']
""")

print('\nAll done!')
