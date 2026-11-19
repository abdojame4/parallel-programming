import sys
from glob import glob
import model_init
from time import perf_counter
from multiprocessing import Pool

image_count = int(sys.argv[1])
core_count = int(sys.argv[2])
images = sorted(glob("data/images/*.jpg"))[:image_count]

if __name__ == "__main__":
    with Pool(core_count, model_init.init_model) as pool:
        pool.map(model_init.detect, images[:core_count])
        t0 = perf_counter()
        pool.map(model_init.detect, images, chunksize=(len(images) // core_count))
        t1 = perf_counter() - t0
        print(t1)