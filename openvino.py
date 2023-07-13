# object detection model
from pathlib import Path
from ultralytics import YOLO

det_model_path = Path("model.xml")
det_model = YOLO('runs/detect/train11/weights/best.pt')
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)
