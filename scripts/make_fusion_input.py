import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')
from model.rsgnet import RSGNet

DEVICE   = 'cuda'
BASE     = '/home/ivlab/carla_ws/fusion_model'
IMG_SIZE = 640

# load trained RSGNet weights
rsgnet = RSGNet(in_ch=5, out_ch=3).to(DEVICE)

# try to load from best checkpoint
ckpt = torch.load(f'{BASE}/checkpoints/best.pt',
                  map_location=DEVICE)
# extract only rsgnet weights
rsg_weights = {k.replace('rsgnet.',''):v
               for k,v in ckpt.items()
               if k.startswith('rsgnet.')}
rsgnet.load_state_dict(rsg_weights)
rsgnet.eval()
print('RSGNet loaded!')

def process_split(split, rgb_dir, voxel_dir, label_dir,
                  out_img_dir, out_lbl_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(Path(rgb_dir).glob('frame_*.png'))
    print(f'{split}: {len(rgb_files)} frames')

    for rgb_path in tqdm(rgb_files, desc=split):
        stem       = rgb_path.stem
        voxel_path = Path(voxel_dir) / f'{stem}.npy'
        label_path = Path(label_dir) / f'{stem}.txt'

        if not voxel_path.exists() or not label_path.exists():
            continue

        # RGB
        rgb = cv2.cvtColor(
            cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        rgb_t = torch.from_numpy(
            rgb.astype(np.float32)/255.0
        ).permute(2,0,1).unsqueeze(0).to(DEVICE)

        # Voxel → SIF
        v = np.load(str(voxel_path)).astype(np.float32)
        bins = [cv2.resize(v[b], (IMG_SIZE, IMG_SIZE))
                for b in range(v.shape[0])]
        voxel_t = torch.from_numpy(
            np.stack(bins)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            sif = rsgnet(voxel_t)  # (1,3,H,W)

        # merge RGB + SIF → 6 channel numpy
        sif_np = sif[0].cpu().numpy()           # (3,H,W)
        sif_np = (sif_np * 255).astype(np.uint8)
        sif_np = sif_np.transpose(1,2,0)         # (H,W,3)

        # save as two separate images with same stem
        # YOLOv8 only takes 3ch, so we save SIF as separate
        # and use RGB for detection, SIF for visualization
        # → actually save RGB only, SIF used in custom model
        cv2.imwrite(str(out_img_dir/f'{stem}.png'),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # save SIF
        sif_dir = out_img_dir.parent / 'sif'
        sif_dir.mkdir(exist_ok=True)
        np.save(str(sif_dir/f'{stem}.npy'),
                sif[0].cpu().numpy())

        # copy label
        shutil.copy(str(label_path),
                    str(out_lbl_dir/f'{stem}.txt'))

process_split(
    'train',
    f'{BASE}/morning/rgb',
    f'{BASE}/morning/voxel',
    f'{BASE}/morning/yolo/labels',
    Path(f'{BASE}/fusion_yolo/images/train'),
    Path(f'{BASE}/fusion_yolo/labels/train'),
)
process_split(
    'val',
    f'{BASE}/night/rgb',
    f'{BASE}/night/voxel',
    f'{BASE}/night/yolo/labels',
    Path(f'{BASE}/fusion_yolo/images/val'),
    Path(f'{BASE}/fusion_yolo/labels/val'),
)
print('Done!')
