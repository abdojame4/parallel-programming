import torch
from ultralytics import YOLO

model = None

def init_model():
    global model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = YOLO("yolo26n.pt")

def detect(image):
    model(image, device="cpu", verbose=False)

if __name__ == "__main__":
    init_model()
    