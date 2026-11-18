from ultralytics import YOLO

model = None

def init_model():

    model = YOLO("yolo26n.pt")
    
    results = model("images")

    for r in results:
        r.show()

if __name__ == "__main__":
    init_model()
    