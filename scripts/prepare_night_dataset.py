import numpy as np
import pandas as pd
import torch
import cv2
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ===== 設定區 =====
BASE_DIR     = '/home/ivlab/carla_ws/fusion_model'
DATASET      = 'night'
B            = 5       # Voxel Grid 時間切片
HEIGHT       = 720
WIDTH        = 1280
MAX_TIME_DIFF = 0.1    # 最大時間差（秒）
BATCH_SIZE   = 4

RGB_DIR      = f'{BASE_DIR}/{DATASET}/rgb'
VOXEL_DIR    = f'{BASE_DIR}/{DATASET}/voxel'
YOLO_LBL_DIR = f'{BASE_DIR}/{DATASET}/yolo/labels'
OUT_DIR      = f'{BASE_DIR}/{DATASET}/aligned'

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 步驟1：對齊 RGB 和 Voxel Grid =====
print("===== 步驟1：對齊 RGB 和 Voxel Grid =====")

rgb_labels = pd.read_csv(f'{BASE_DIR}/{DATASET}/rgb_labels.csv')
dvs_labels = pd.read_csv(f'{BASE_DIR}/{DATASET}/dvs_labels.csv')

print(f"RGB frames：{len(rgb_labels)}")
print(f"DVS frames：{len(dvs_labels)}")

pairs = []
for _, rgb_row in tqdm(rgb_labels.iterrows(), total=len(rgb_labels)):
    rgb_ts = rgb_row['timestamp']

    # 找最近的 DVS timestamp
    diff        = (dvs_labels['timestamp'] - rgb_ts).abs()
    closest_idx = diff.idxmin()
    dvs_row     = dvs_labels.iloc[closest_idx]
    time_diff   = abs(rgb_ts - dvs_row['timestamp'])

    if time_diff > MAX_TIME_DIFF:
        continue

    # 確認檔案都存在
    rgb_file   = f'{RGB_DIR}/frame_{int(rgb_row["frame"]):06d}.png'
    voxel_file = f'{VOXEL_DIR}/frame_{int(dvs_row["frame"]):06d}.npy'
    label_file = f'{YOLO_LBL_DIR}/frame_{int(rgb_row["frame"]):06d}.txt'

    if not os.path.exists(rgb_file):
        continue
    if not os.path.exists(voxel_file):
        continue

    pairs.append({
        'rgb_frame':   int(rgb_row['frame']),
        'dvs_frame':   int(dvs_row['frame']),
        'rgb_file':    rgb_file,
        'voxel_file':  voxel_file,
        'label_file':  label_file,
        'time_diff':   time_diff,
        'throttle':    rgb_row['throttle'],
        'brake':       rgb_row['brake'],
        'steer':       rgb_row['steer'],
        'speed':       rgb_row['speed'],
    })

pairs_df = pd.DataFrame(pairs)
print(f"\n成功配對：{len(pairs_df)} 組")
print(f"平均時間差：{pairs_df['time_diff'].mean()*1000:.2f} ms")
print(f"丟棄：{len(rgb_labels) - len(pairs_df)} 組")

# 存配對結果
pairs_df.to_csv(f'{OUT_DIR}/pairs.csv', index=False)
print(f"✅ 配對結果存在 {OUT_DIR}/pairs.csv")

# ===== 步驟2：切分訓練集和驗證集 =====
print("\n===== 步驟2：切分訓練集和驗證集 =====")

train_df, val_df = train_test_split(
    pairs_df,
    test_size=0.2,
    random_state=42
)

train_df.to_csv(f'{OUT_DIR}/train_pairs.csv', index=False)
val_df.to_csv(f'{OUT_DIR}/val_pairs.csv', index=False)

print(f"訓練集：{len(train_df)} 組")
print(f"驗證集：{len(val_df)} 組")
print(f"✅ 存在 {OUT_DIR}/")

# ===== 步驟3：定義 PyTorch Dataset =====
print("\n===== 步驟3：定義 PyTorch Dataset =====")

class NightFusionDataset(Dataset):
    def __init__(self, pairs_df, height=720, width=1280, B=5):
        self.pairs  = pairs_df.reset_index(drop=True)
        self.height = height
        self.width  = width
        self.B      = B

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        # ===== 讀取 RGB =====
        rgb = cv2.imread(row['rgb_file'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height))
        # 轉成 Tensor (3, H, W)，正規化到 0~1
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # ===== 讀取 Voxel Grid =====
        # float16 → float32（PyTorch 需要 float32）
        voxel = np.load(row['voxel_file']).astype(np.float32)
        # 確認形狀正確 (5, H, W)
        if voxel.shape != (self.B, self.height, self.width):
            voxel = np.resize(voxel, (self.B, self.height, self.width))
        # 轉成 Tensor (5, H, W)
        voxel_tensor = torch.from_numpy(voxel).float()

        # ===== 讀取 YOLO 標註 =====
        boxes = []
        if os.path.exists(row['label_file']):
            with open(row['label_file'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(x) for x in parts])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'rgb':      rgb_tensor,    # (3, 720, 1280)
            'voxel':    voxel_tensor,  # (5, 720, 1280)
            'boxes':    boxes_tensor,  # (N, 5)
            'throttle': torch.tensor(row['throttle'], dtype=torch.float32),
            'brake':    torch.tensor(row['brake'],    dtype=torch.float32),
            'steer':    torch.tensor(row['steer'],    dtype=torch.float32),
        }

# ===== 步驟4：測試 Dataset =====
print("\n===== 步驟4：測試 Dataset =====")

train_dataset = NightFusionDataset(train_df, HEIGHT, WIDTH, B)
val_dataset   = NightFusionDataset(val_df,   HEIGHT, WIDTH, B)

# 測試讀取一筆數據
sample = train_dataset[0]
print(f"RGB tensor 形狀：    {sample['rgb'].shape}")
print(f"Voxel tensor 形狀：  {sample['voxel'].shape}")
print(f"Boxes 形狀：         {sample['boxes'].shape}")
print(f"Throttle：           {sample['throttle']}")
print(f"Brake：              {sample['brake']}")
print(f"Steer：              {sample['steer']}")

# ===== 步驟5：建立 DataLoader =====
print("\n===== 步驟5：建立 DataLoader =====")

def collate_fn(batch):
    rgb    = torch.stack([b['rgb']   for b in batch])
    voxel  = torch.stack([b['voxel'] for b in batch])
    boxes  = [b['boxes'] for b in batch]  # 每張圖 boxes 數量不同
    throttle = torch.stack([b['throttle'] for b in batch])
    brake    = torch.stack([b['brake']    for b in batch])
    steer    = torch.stack([b['steer']    for b in batch])
    return {
        'rgb':      rgb,
        'voxel':    voxel,
        'boxes':    boxes,
        'throttle': throttle,
        'brake':    brake,
        'steer':    steer,
    }

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    collate_fn  = collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 2,
    collate_fn  = collate_fn
)

print(f"訓練 DataLoader：{len(train_loader)} 個 batch")
print(f"驗證 DataLoader：{len(val_loader)} 個 batch")

# ===== 存 Dataset =====
import pickle
with open(f'{BASE_DIR}/train_night_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open(f'{BASE_DIR}/val_night_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)

print(f"\n✅ 全部完成！")
print(f"   配對：{len(pairs_df)} 組")
print(f"   訓練集：{len(train_df)} 組")
print(f"   驗證集：{len(val_df)} 組")
print(f"   RGB tensor：(3, {HEIGHT}, {WIDTH})")
print(f"   Voxel tensor：({B}, {HEIGHT}, {WIDTH})")