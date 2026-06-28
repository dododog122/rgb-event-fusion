import os
import glob
import random

BASE      = '/home/ivlab/carla_ws/fusion_model/morning/yolo'
TRAIN_IMG = f'{BASE}/train/images'
TRAIN_LBL = f'{BASE}/train/labels'
VAL_IMG   = f'{BASE}/val/images'
VAL_LBL   = f'{BASE}/val/labels'

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(VAL_IMG,   exist_ok=True)
os.makedirs(VAL_LBL,   exist_ok=True)

files = sorted(glob.glob(f'{BASE}/images/frame_*.png'))
random.seed(42)
random.shuffle(files)

split       = int(len(files) * 0.8)
train_files = files[:split]
val_files   = files[split:]

print(f"訓練集：{len(train_files)} 張")
print(f"驗證集：{len(val_files)} 張")

for f in train_files:
    name = os.path.basename(f)
    stem = name.replace('.png', '')
    # 用軟連結，不複製檔案
    dst_img = f'{TRAIN_IMG}/{name}'
    dst_lbl = f'{TRAIN_LBL}/{stem}.txt'
    if not os.path.exists(dst_img):
        os.symlink(f, dst_img)
    lbl = f'{BASE}/labels/{stem}.txt'
    if os.path.exists(lbl) and not os.path.exists(dst_lbl):
        os.symlink(lbl, dst_lbl)

for f in val_files:
    name = os.path.basename(f)
    stem = name.replace('.png', '')
    dst_img = f'{VAL_IMG}/{name}'
    dst_lbl = f'{VAL_LBL}/{stem}.txt'
    if not os.path.exists(dst_img):
        os.symlink(f, dst_img)
    lbl = f'{BASE}/labels/{stem}.txt'
    if os.path.exists(lbl) and not os.path.exists(dst_lbl):
        os.symlink(lbl, dst_lbl)

print("✅ 切分完成！（使用軟連結，不佔額外空間）")