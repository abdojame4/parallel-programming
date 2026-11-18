import json
import random
import requests
from pathlib import Path
from ultralytics.data.converter import convert_coco

with open("dataset/annotations/instances_val2017.json", "r", encoding="utf-8") as file_data:
    coco_annotations = json.load(file_data)

image_annotations = {}
for anno in coco_annotations["annotations"]:
    image_annotations.setdefault(anno["image_id"], []).append(anno)

images = [img for img in coco_annotations["images"] if img["id"] in image_annotations]

random.seed(0)
random.shuffle(images)
subset = images[:500]

subset_id = [img["id"] for img in subset]
subset_annotations = [anno for anno in coco_annotations["annotations"] if anno["image_id"] in subset_id]

subset_json = {
    "info": coco_annotations.get("info", {}),
    "licenses": coco_annotations.get("licenses", {}),
    "images": subset,
    "annotations": subset_annotations,
    "categories": coco_annotations["categories"]
}
if not Path("dataset/annotations/subset_val.json").is_file:
    with open("dataset/annotations/subset_val.json", "w") as file:
        json.dump(subset_json, file)

for img in subset:
    file_name = img["file_name"]
    file_path = f"dataset/images/val/{file_name}"
    if not Path(file_path).is_file():
        with open(file_path, "wb") as image_file:
            image_file.write(requests.get(f"http://images.cocodataset.org/val2017/{file_name}").content)

convert_coco(labels_dir="dataset/annotations/", save_dir="dataset", use_segments=False, cls91to80=False)