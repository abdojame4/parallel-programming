import torch
from ultralytics import YOLO

model = None

def init_model(model_path="yolo26n.pt"):
    global model
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = YOLO(model_path)

def detect(image):
    results = model(
        image,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False,
        save=False,
    )
    return len(results[0].boxes)

if __name__ == "__main__":
    init_model()
    