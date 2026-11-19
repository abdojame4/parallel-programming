
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

## Benchmarking

Run the sequential baseline and multiprocessing benchmarks on the first 100,
200, and 500 sorted images in `dataset/images/val`:

```sh
python parallel_timing.py --sizes 100 200 500 --workers 1 2 4 --repeats 3
```

The script uses `yolo26n.pt`, CPU inference, `imgsz=640`, and `conf=0.25` for
both sequential and parallel runs. Total execution time includes model loading
and, for multiprocessing, worker startup. It writes individual runs to
`benchmark_results.csv` and aggregate speedup/efficiency metrics to
`benchmark_summary.csv`. Existing output files are not overwritten unless
`--overwrite` is passed.

These results reflect the machine's current CPU configuration. For Docker CPU
limit experiments, run the same command inside each agreed container limit and
save each run to separate CSV paths with `--results` and `--summary`.