from ultralytics import YOLO
import cv2
import os
import glob
import shutil
from tqdm import tqdm

# ===== 設定區 =====
DATASETS = ['morning', 'night']
# YOLOv8 預訓練模型（自動下載）
model = YOLO('yolov8n.pt')

# COCO 類別對應到你的類別
# COCO: 2=car, 5=bus, 7=truck, 0=person, 9=traffic light, 11=stop sign
COCO_TO_CUSTOM = {
    2:  0,   # car       → 車輛
    5:  0,   # bus       → 車輛
    7:  0,   # truck     → 車輛
    0:  1,   # person    → 行人
    9:  3,   # traffic light → 紅綠燈
    11: 2,   # stop sign → 交通標誌
}

for dataset in DATASETS:
    print(f"\n===== 處理 {dataset} =====")

    RGB_DIR  = f'/home/ivlab/carla_ws/fusion_model/{dataset}/rgb'
    YOLO_IMG = f'/home/ivlab/carla_ws/fusion_model/{dataset}/yolo/images'
    YOLO_LBL = f'/home/ivlab/carla_ws/fusion_model/{dataset}/yolo/labels'

    os.makedirs(YOLO_IMG, exist_ok=True)
    os.makedirs(YOLO_LBL, exist_ok=True)

    files = sorted(glob.glob(f'{RGB_DIR}/frame_*.png'))
    print(f"找到 {len(files)} 張圖片")

    for filepath in tqdm(files):
        filename = os.path.basename(filepath)
        stem     = filename.replace('.png', '')

        # 用 YOLOv8 偵測
        results = model(filepath, verbose=False)[0]

        shutil.copy(filepath, f'{YOLO_IMG}/{filename}')

        # 存 YOLO 格式標註
        with open(f'{YOLO_LBL}/{stem}.txt', 'w') as f:
            for box in results.boxes:
                coco_class = int(box.cls)
                if coco_class not in COCO_TO_CUSTOM:
                    continue
                custom_class = COCO_TO_CUSTOM[coco_class]
                x_c, y_c, w, h = box.xywhn[0].tolist()
                f.write(f"{custom_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ {dataset} 標註完成！")

print("\n===== 全部完成！=====")