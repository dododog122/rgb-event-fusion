from ultralytics import YOLO

# 載入預訓練模型
model = YOLO('yolov8n.pt')

# 開始訓練
model.train(
    data    = '/home/ivlab/carla_ws/fusion_model/dataset.yaml',
    epochs  = 50,
    imgsz   = 640,
    batch   = 8,
    device  = 0,        # 用 GPU
    project = '/home/ivlab/carla_ws/fusion_model/runs',
    name    = 'rgb_baseline',
    save    = True,
)

print("✅ 訓練完成！")