import sys
from glob import glob
import model_init
from time import perf_counter

image_count = int(sys.argv[1])
images = sorted(glob("data/images/*.jpg"))[:image_count]

if __name__ == "__main__":
    model_init.init_model()
    model_init.detect(images[0])
    t0 = perf_counter()
    for img in images:
        model_init.detect(img)
    t1 = perf_counter() - t0
    print(t1)