from ultralytics import YOLO

# Load YOLO11m model
model = YOLO("yolo11m.pt")

# Train
model.train(
    data="/zhome/e5/7/219270/Perception_AutonomousSystems/YOLO_data.yaml",   # your dataset config
    epochs=100,         # training epochs
    imgsz=640,          # training input size
    batch=2,           # batch size (adjust based on GPU RAM)
    device=0,           # GPU (set 'cpu' if no GPU)
    workers=2,          # data loader workers
    # optimizer="SGD",    # or 'AdamW'
)
