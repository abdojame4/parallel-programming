import json
import random
from pathlib import Path

import requests
from ultralytics.data.converter import convert_coco

annotations_dir = Path("dataset/annotations")
images_dir = Path("dataset/images/val")
source_path = annotations_dir / "instances_val2017.json"
subset_path = annotations_dir / "subset_val.json"

with source_path.open(encoding="utf-8") as file_data:
    coco_annotations = json.load(file_data)

image_annotations = {}
for anno in coco_annotations["annotations"]:
    image_annotations.setdefault(anno["image_id"], []).append(anno)

images = [img for img in coco_annotations["images"] if img["id"] in image_annotations]

if subset_path.is_file():
    with subset_path.open(encoding="utf-8") as file_data:
        subset_json = json.load(file_data)
else:
    random.seed(0)
    random.shuffle(images)
    subset = images[:500]
    subset_ids = {img["id"] for img in subset}
    subset_json = {
        "info": coco_annotations.get("info", {}),
        "licenses": coco_annotations.get("licenses", []),
        "images": subset,
        "annotations": [
            anno for anno in coco_annotations["annotations"]
            if anno["image_id"] in subset_ids
        ],
        "categories": coco_annotations["categories"],
    }
    with subset_path.open("w", encoding="utf-8") as file_data:
        json.dump(subset_json, file_data)

subset = subset_json["images"]
subset_ids = [img["id"] for img in subset]
subset_id_set = set(subset_ids)
source_images = {img["id"]: img for img in coco_annotations["images"]}
subset_annotation_ids = {
    anno["image_id"] for anno in subset_json["annotations"]
}

if len(subset_ids) != 500 or len(subset_id_set) != 500:
    raise ValueError("subset_val.json must contain exactly 500 unique images")
if not subset_id_set <= source_images.keys():
    raise ValueError("subset_val.json contains image IDs missing from source annotations")
if any(source_images[img["id"]]["file_name"] != img["file_name"] for img in subset):
    raise ValueError("subset_val.json contains filenames that do not match source annotations")
if not subset_annotation_ids <= subset_id_set or not subset_id_set <= subset_annotation_ids:
    raise ValueError("subset_val.json annotations do not match its 500 image IDs")

images_dir.mkdir(parents=True, exist_ok=True)
for img in subset:
    file_name = img["file_name"]
    file_path = images_dir / file_name
    if not file_path.is_file():
        response = requests.get(
            f"http://images.cocodataset.org/val2017/{file_name}",
            timeout=30,
        )
        response.raise_for_status()
        file_path.write_bytes(response.content)

convert_coco(labels_dir="dataset/annotations/", save_dir="dataset", use_segments=False, cls91to80=False)