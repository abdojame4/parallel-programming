import json
import random
import sys
import requests
from pathlib import Path

with open("data/annotations/instances_val2017.json", "r", encoding="utf-8") as file_data:
    coco_annotations = json.load(file_data)

images = [img for img in coco_annotations["images"]]

random.seed(0)
random.shuffle(images)
image_count = int(sys.argv[1])
subset = images[:image_count]

for img in subset:
    file_name = img["file_name"]
    file_path = f"data/images/{file_name}"
    if not Path(file_path).is_file():
        with open(file_path, "wb") as image_file:
            image_file.write(requests.get(f"http://images.cocodataset.org/val2017/{file_name}").content)

