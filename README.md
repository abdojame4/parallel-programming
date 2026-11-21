
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

### run test
```sh
python sequential.py <image-count>
python parallel.py <image-count> <core-count>
```

### run benchmarks

```sh
python parallel_timing.py --sizes 100 200 500 --workers 1 2 4 --repeats 3
```

The benchmark script compares sequential and multiprocessing YOLO inference on
CPU and writes raw runs to `benchmark_results.csv` and aggregate metrics to
`benchmark_summary.csv`.