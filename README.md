
# parallel-programming

group project

## setup

### python virtual environment

```sh
python -m venv .venv
source .venv/bin/activate # macos and linux
.venv\Scripts\activate.ps1 # windows
```

### dependencies

```sh
pip install -r requirements.txt
```

### download dataset

```sh
python extract.py <image-count> # default 500, max 5000
```

### run detection

`sequential.py` processes the requested number of sorted images from
`data/images/`.
`parallel.py` processes all `.jpg`, `.jpeg`, and `.png` files there.

```sh
python sequential.py
python sequential.py --image-count 25
python parallel.py --workers 2
```

Sequential processing defaults to 10 images; use `--image-count` to change it.
The parallel command's `--workers` option defaults to 2. Both scripts use
`yolo26n.pt` on CPU with `imgsz=640` and `conf=0.25`, as configured in
`model_init.py`.

### run benchmarks

```sh
python parallel_timing.py --sizes 100 200 500 --workers 1 2 4 --repeats 3
```

The benchmark script compares sequential and multiprocessing YOLO inference on
CPU and writes raw runs to `benchmark_results.csv` and aggregate metrics to
`benchmark_summary.csv`.