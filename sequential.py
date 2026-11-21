import argparse
from pathlib import Path
import model_init
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description="Run sequential YOLO detection.")
    parser.add_argument(
        "--image-count",
        type=int,
        default=10,
        help="number of images to process (default: 10)",
    )
    args = parser.parse_args()

    if args.image_count < 1:
        parser.error("--image-count must be at least 1")

    image_dir = Path("data/images")
    if not image_dir.is_dir():
        parser.error(f"image directory not found: {image_dir}")

    available_images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(available_images) < args.image_count:
        parser.error(
            f"requested {args.image_count} images, but only "
            f"{len(available_images)} are available in {image_dir}"
        )
    images = available_images[:args.image_count]

    model_init.init_model()
    start = perf_counter()
    for image in images:
        detections = model_init.detect(str(image))
        print(f"{image.name}: {detections} objects detected")
    elapsed = perf_counter() - start

    print(
        f"Processed {len(images)} images sequentially "
        f"in {elapsed:.2f} seconds."
    )


if __name__ == "__main__":
    main()