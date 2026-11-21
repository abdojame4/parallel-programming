import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import model_init


def main():
    parser = argparse.ArgumentParser(description="Run parallel YOLO detection.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    image_dir = Path("data/images")
    if not image_dir.is_dir():
        parser.error(f"image directory not found: {image_dir}")

    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Images: {len(images)}")
    print(f"Workers: {args.workers}")
    if not images:
        print("No images found.")
        return

    start = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=args.workers,
        initializer=model_init.init_model,
    ) as pool:
        counts = pool.map(model_init.detect, images, chunksize=1)
    elapsed = time.perf_counter() - start

    for image, count in zip(images, counts):
        print(f"{image.name}: {count} objects")
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()