from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data = "animals_dataset.yaml",epochs = 100, save_period = 10)