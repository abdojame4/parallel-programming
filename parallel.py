import argparse
import multiprocessing as mp
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from glob import glob
import model_init
from time import perf_counter
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Run parallel YOLO detection.")
parser.add_argument("image_count", type=int)
parser.add_argument("core_count", type=int)
args = parser.parse_args()

if args.image_count < 1:
    parser.error("image_count must be at least 1")
if args.core_count < 1:
    parser.error("core_count must be at least 1")

images = sorted(glob("data/images/*.jpg"))[:args.image_count]

if __name__ == "__main__":
    context = mp.get_context("spawn")
    with context.Pool(args.core_count, model_init.init_model) as pool:
        t0 = perf_counter()
        counts = pool.map(model_init.detect, images, chunksize=1)
        t1 = perf_counter() - t0
        for image, count in zip(images, counts):
            print(f"{os.path.basename(image)}: {count} objects")
        print(t1)