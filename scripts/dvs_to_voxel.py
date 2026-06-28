import numpy as np
import os
import glob
from tqdm import tqdm

# ===== 設定區 =====
DATASETS  = ['morning', 'night']
B         = 5      # 時間切片數量
HEIGHT    = 720
WIDTH     = 1280

def events_to_voxel_grid(events, B, H, W):
    """
    把 events 轉成 Voxel Grid
    輸出形狀：(B, H, W)
    """
    voxel = np.zeros((B, H, W), dtype=np.float16)

    if len(events) == 0:
        return voxel

    # 取得時間範圍
    t_min = events['t'].min()
    t_max = events['t'].max()
    t_range = t_max - t_min

    if t_range == 0:
        return voxel

    # 每個 event 對應到哪個切片
    # 正極 = +1，負極 = -1
    for e in events:
        # 計算切片索引
        t_norm = (e['t'] - t_min) / t_range  # 0~1
        b_idx  = min(int(t_norm * B), B - 1)  # 0~B-1

        x = int(e['x'])
        y = int(e['y'])

        if 0 <= x < W and 0 <= y < H:
            if e['pol']:
                voxel[b_idx, y, x] += 1.0   # 正極
            else:
                voxel[b_idx, y, x] -= 1.0   # 負極

    # 正規化到 -1 ~ 1
    max_val = np.abs(voxel).max()
    if max_val > 0:
        voxel = voxel / max_val

    return voxel

# ===== 轉換 =====
for dataset in DATASETS:
    print(f"\n===== 處理 {dataset} =====")

    DVS_DIR   = f'/home/ivlab/carla_ws/fusion_model/{dataset}/dvs'
    VOXEL_DIR = f'/home/ivlab/carla_ws/fusion_model/{dataset}/voxel'
    os.makedirs(VOXEL_DIR, exist_ok=True)

    files = sorted(glob.glob(f'{DVS_DIR}/frame_*.npy'))
    print(f"找到 {len(files)} 個 DVS 檔案")

    for filepath in tqdm(files):
        filename = os.path.basename(filepath)

        # 讀取 events
        events = np.load(filepath)

        # 轉換成 Voxel Grid
        voxel = events_to_voxel_grid(events, B, HEIGHT, WIDTH)

        # 存成 npy（形狀：(20, 720, 1280)）
        out_path = os.path.join(VOXEL_DIR, filename)
        np.save(out_path, voxel)

    print(f"✅ {dataset} 完成！")
    print(f"   Voxel Grid 形狀：({B}, {HEIGHT}, {WIDTH})")
    print(f"   存在：{VOXEL_DIR}")

print("\n===== 全部完成！=====")