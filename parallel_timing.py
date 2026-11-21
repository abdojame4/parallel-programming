import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import model_init

model = None
model_loading_seconds = 0.0
IMAGE_SIZE = 640
CONFIDENCE = 0.25
DEVICE = "cpu"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_model(model_path):
    start = time.perf_counter()
    model_init.init_model(model_path)
    return time.perf_counter() - start


def init_worker(model_path):
    global model, model_loading_seconds
    model_loading_seconds = load_model(model_path)


def detect_image(image_path):
    start = time.perf_counter()
    count = model_init.detect(str(image_path))
    elapsed = time.perf_counter() - start
    return (
        str(image_path),
        count,
        elapsed,
        os.getpid(),
        model_loading_seconds,
    )


def run_sequential(images, model_path):
    global model, model_loading_seconds
    total_start = time.perf_counter()
    model_loading_seconds = load_model(model_path)
    results = [detect_image(image_path) for image_path in images]
    return results, time.perf_counter() - total_start


def run_parallel(images, model_path, workers):
    total_start = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(model_path,),
    ) as pool:
        results = pool.map(detect_image, images, chunksize=1)
    return results, time.perf_counter() - total_start


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file_data:
        writer = csv.DictWriter(file_data, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequential and multiprocessing YOLO inference."
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--results", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("benchmark_summary.csv"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing result files",
    )
    args = parser.parse_args()

    existing_outputs = [
        path for path in (args.results, args.summary) if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        parser.error(
            "output file(s) already exist: "
            + ", ".join(str(path) for path in existing_outputs)
            + "; choose different paths or pass --overwrite"
        )

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if any(workers < 1 for workers in args.workers):
        parser.error("worker counts must be at least 1")
    if any(size < 1 for size in args.sizes):
        parser.error("image counts must be at least 1")
    if not Path(args.model).is_file():
        parser.error(f"model file not found: {args.model}")
    if not args.image_dir.is_dir():
        parser.error(f"image directory not found: {args.image_dir}")

    all_images = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not all_images:
        parser.error(f"no supported images found in {args.image_dir}")
    if max(args.sizes) > len(all_images):
        parser.error(
            f"requested {max(args.sizes)} images, but only {len(all_images)} are available"
        )

    raw_fields = [
        "image_count",
        "mode",
        "workers",
        "repeat",
        "images_processed",
        "total_execution_seconds",
        "sum_detection_seconds",
        "average_detection_seconds",
        "median_detection_seconds",
        "max_model_loading_seconds",
        "throughput_images_per_second",
    ]
    raw_rows = []
    for size in args.sizes:
        images = all_images[:size]
        configurations = [("sequential", 1)]
        configurations.extend(("parallel", count) for count in args.workers)

        for mode, workers in configurations:
            for repeat in range(1, args.repeats + 1):
                if mode == "sequential":
                    results, total_seconds = run_sequential(images, args.model)
                else:
                    results, total_seconds = run_parallel(
                        images, args.model, workers
                    )

                detection_times = [result[2] for result in results]
                load_times = {
                    result[3]: result[4] for result in results
                }
                row = {
                    "image_count": size,
                    "mode": mode,
                    "workers": workers,
                    "repeat": repeat,
                    "images_processed": len(results),
                    "total_execution_seconds": f"{total_seconds:.6f}",
                    "sum_detection_seconds": f"{sum(detection_times):.6f}",
                    "average_detection_seconds": (
                        f"{statistics.mean(detection_times):.6f}"
                    ),
                    "median_detection_seconds": (
                        f"{statistics.median(detection_times):.6f}"
                    ),
                    "max_model_loading_seconds": (
                        f"{max(load_times.values()):.6f}"
                    ),
                    "throughput_images_per_second": (
                        f"{len(results) / total_seconds:.6f}"
                    ),
                }
                raw_rows.append(row)
                write_rows(args.results, raw_fields, raw_rows)
                print(
                    f"{size} images | {mode} | workers={workers} | "
                    f"run {repeat}/{args.repeats} | "
                    f"{total_seconds:.3f} s | "
                    f"{len(results)} images processed",
                    flush=True,
                )

    summary_fields = [
        "image_count",
        "mode",
        "workers",
        "runs",
        "average_total_seconds",
        "median_total_seconds",
        "average_throughput_images_per_second",
        "speedup_vs_sequential",
        "efficiency_percent",
    ]
    summary_rows = []
    for size in args.sizes:
        size_rows = [
            row for row in raw_rows
            if int(row["image_count"]) == size
        ]
        sequential_rows = [
            row for row in size_rows if row["mode"] == "sequential"
        ]
        if not sequential_rows:
            raise RuntimeError(f"no sequential baseline recorded for {size} images")
        sequential_average = statistics.mean(
            float(row["total_execution_seconds"]) for row in sequential_rows
        )

        groups = {}
        for row in size_rows:
            key = (row["mode"], int(row["workers"]))
            groups.setdefault(key, []).append(row)

        for (mode, workers), rows in sorted(groups.items()):
            average_seconds = statistics.mean(
                float(row["total_execution_seconds"]) for row in rows
            )
            speedup = sequential_average / average_seconds
            summary_rows.append({
                "image_count": size,
                "mode": mode,
                "workers": workers,
                "runs": len(rows),
                "average_total_seconds": f"{average_seconds:.6f}",
                "median_total_seconds": f"{statistics.median(float(row['total_execution_seconds']) for row in rows):.6f}",
                "average_throughput_images_per_second": f"{statistics.mean(float(row['throughput_images_per_second']) for row in rows):.6f}",
                "speedup_vs_sequential": f"{speedup:.6f}",
                "efficiency_percent": (
                    "" if mode == "sequential"
                    else f"{speedup / workers * 100:.2f}"
                ),
            })

    write_rows(args.summary, summary_fields, summary_rows)
    print(f"Raw results: {args.results}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
