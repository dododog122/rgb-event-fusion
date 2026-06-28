import sys
import os
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from v4.model.event_transformer import EventMAETransformer
from v4.model.fusion import CrossModalCBAM

DEVICE   = 'cuda'
BASE     = '/home/ivlab/carla_ws/fusion_model'
CKPT_DIR = Path(f'{BASE}/checkpoints')
IMG_SIZE = 640


# ── Step 1: 用 event encoder 生成 attention-weighted fusion 圖片 ──
# 這次用更好的 fusion：
# attention map 指導哪些區域要更重視 event
# 然後在 feature map 層面做融合

def generate_v4_fusion_images(split, rgb_dir, voxel_dir,
                               label_dir, out_img, out_lbl,
                               encoder, device):
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)

    files = sorted(Path(rgb_dir).glob('frame_*.png'))
    print(f'{split}: {len(files)} frames')

    encoder.eval()
    with torch.no_grad():
        for rgb_path in tqdm(files, desc=split):
            stem = rgb_path.stem
            vp   = Path(voxel_dir)/f'{stem}.npy'
            lp   = Path(label_dir)/f'{stem}.txt'
            if not vp.exists() or not lp.exists():
                continue

            # RGB with CLAHE
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            rgb_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Event voxel
            v    = np.load(str(vp)).astype(np.float32)
            bins = [cv2.resize(v[b], (IMG_SIZE,IMG_SIZE))
                    for b in range(v.shape[0])]
            vt   = torch.from_numpy(
                np.stack(bins)).unsqueeze(0).to(device)

            # Get attention map from encoder
            tokens = encoder(vt, pretrain=False)  # (1,1600,256)

            # token norm → spatial attention
            attn = tokens[0].norm(dim=-1)          # (1600,)
            attn = attn.reshape(40, 40).cpu().numpy()
            attn = cv2.resize(attn, (IMG_SIZE, IMG_SIZE))
            attn = (attn - attn.min()) / (attn.max()-attn.min()+1e-6)

            # Event polarity
            vsum = v.sum(0)
            vsum = cv2.resize(vsum, (IMG_SIZE, IMG_SIZE))
            pos  = np.clip( vsum*5, 0, 1)*255
            neg  = np.clip(-vsum*5, 0, 1)*255

            # Attention-weighted event
            evt = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.float32)
            evt[:,:,1] = pos * attn
            evt[:,:,2] = neg * attn

            # Blend
            fused = cv2.addWeighted(
                rgb_enh.astype(np.float32), 0.65,
                evt, 0.35, 0).clip(0,255).astype(np.uint8)

            cv2.imwrite(str(Path(out_img)/f'{stem}.png'), fused)

            import shutil
            shutil.copy(str(lp), str(Path(out_lbl)/f'{stem}.txt'))


# ── Step 2: 用 ultralytics YOLOv8 訓練 fusion 圖片 ──
def train_yolo_on_fusion():
    # Load encoder
    encoder = EventMAETransformer(
        in_ch=5, img_size=640, patch_size=16,
        embed_dim=256, depth=6, n_heads=8,
        mask_ratio=0.75
    ).to(DEVICE)
    encoder.load_state_dict(
        torch.load(CKPT_DIR/'v4_event_encoder.pt',
                   map_location=DEVICE))
    print('Event encoder loaded!')

    # Generate fusion images
    out_base = f'{BASE}/fusion_yolo_v4b'
    generate_v4_fusion_images(
        'train',
        f'{BASE}/morning/rgb', f'{BASE}/morning/voxel',
        f'{BASE}/morning/yolo/labels',
        f'{out_base}/images/train',
        f'{out_base}/labels/train',
        encoder, DEVICE)

    generate_v4_fusion_images(
        'val',
        f'{BASE}/night/rgb', f'{BASE}/night/voxel',
        f'{BASE}/night/yolo/labels',
        f'{out_base}/images/val',
        f'{out_base}/labels/val',
        encoder, DEVICE)

    # Write data yaml
    yaml_path = f'{out_base}/data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {out_base}
train: images/train
val:   images/val
nc: 4
names: ['vehicle','pedestrian','traffic_sign','traffic_light']
""")
    print(f'Data yaml: {yaml_path}')

    # Train with ultralytics — full YOLOv8 loss + mAP
    model = YOLO('yolov8n.pt')
    model.train(
        data    = yaml_path,
        epochs  = 50,
        imgsz   = IMG_SIZE,
        batch   = 8,
        device  = 0,
        project = 'runs',
        name    = 'fusion_yolov8_v4b',
        exist_ok= True,
        workers = 0,
    )
    print('Training done!')
    print('Results saved to runs/detect/fusion_yolov8_v4b/')


if __name__ == '__main__':
    train_yolo_on_fusion()
