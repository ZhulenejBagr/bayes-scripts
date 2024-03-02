# bayes-scripts
various bayes scripts


# Install
1. Create image:
```
cd docker; ./build-image.sh; cd ..
```
Note: *`docker/requirements.txt` is used to install packages into the image*
2. Create virtual environment using the container:
```
./bin/fterm
./bin/setup_venv
```
Note 1: *`./requirements.txt` is used to install packages into the venv*  
Note 2: *Different packages required for `pymc` and `tinyDA` due to different
        `arviz` version dependency*  
Note 3: *Currently, we do not even need our own image. All packages go into `venv`.*


# Run tests
Open container, activate venv, run pytest:
```
./bin/fterm
source venv/bin/activate
cd tests/common
pytest test_common.py
```

# Run tinyDA sample script
Open container, activate venv, run script:
```
./bin/fterm
source venv/bin/activate
./bin/run_tinyda_sample.sh

```