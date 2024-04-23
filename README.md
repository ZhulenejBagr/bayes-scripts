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
# Setup Singularity environment
Build image from existing docker image and enter it:
```
./bin/fterm_sing
exit
```

# Run MLDA sampling on Metacentrum
Log into Metacentrum, cd to repo:

(Change params in ./scripts/sample_template/config.yaml)
```
qsub ./scripts/pbs_singularity_shell.sh
```