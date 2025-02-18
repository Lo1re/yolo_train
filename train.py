from ultralytics import YOLO

# Завантажуємо модель YOLOv8 (архітектура)
model = YOLO("yolov8n.pt")  # Використовуємо початкову модель

# Навчання
model.train(data="config.yaml", epochs=50, imgsz=640, batch=16, device="cuda")  # Використання GPU
