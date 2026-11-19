
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
python extract.py
```

### run test
```sh
python sequential.py <image-count>
python parallel.py <image-count> <core-count>
```