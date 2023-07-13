from ultralytics import YOLO

model = YOLO("yolov8m.yaml")

results = model("Assets/dog_walk.mp4", save = True, project = "Results/")

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Class probabilities for classification outputs