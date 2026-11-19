
import os

# Limit CPU threads before importing YOLO/PyTorch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from ultralytics import YOLO

model = None


def init_worker():
    global model

    import torch
    torch.set_num_threads(1)

    # Each worker loads YOLO only once.
    model = YOLO("yolo26n.pt")


def detect_image(image_path):
    global model

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False,
        save=False
    )

    count = len(results[0].boxes)
    return str(image_path), count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("workers must be at least 1")

    images = sorted(
        p for p in Path("dataset/images/val").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    print(f"Images: {len(images)}")
    print(f"Workers: {args.workers}")

    if not images:
        print("No images found.")
        return

    start = time.perf_counter()

    # Spawn is compatible with macOS.
    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=args.workers,
        initializer=init_worker
    ) as pool:
        results = pool.map(detect_image, images)

    elapsed = time.perf_counter() - start

    for path, count in results:
        print(f"{Path(path).name}: {count} objects")

    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()


