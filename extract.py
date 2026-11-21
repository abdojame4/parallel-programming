import json
import random
import argparse
import requests
from pathlib import Path

parser = argparse.ArgumentParser(description="Download a subset of COCO validation images.")
parser.add_argument("image_count", nargs="?", type=int, default=500,
                    help="number of images to download (default: 500)")
args = parser.parse_args()

if args.image_count < 1 or args.image_count > 5000:
    parser.error("image_count must be between 1 and 5000")

with open("data/annotations/instances_val2017.json", "r", encoding="utf-8") as file_data:
    coco_annotations = json.load(file_data)

images = [img for img in coco_annotations["images"]]

random.seed(0)
random.shuffle(images)
subset = images[:args.image_count]

images_dir = Path("data/images")
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

