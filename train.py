from ultralytics import YOLO

# Load pretrained YOLO11
model = YOLO("yolo11n.pt")

# Training menggunakan dataset mangga lokal
results = model.train(
    data="datasets/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    project="runs/mango",
    name="yolo11n_mango",
    exist_ok=True
)
