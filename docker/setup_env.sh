#!/bin/bash
cd /
# PYMC env
mkdir /mnt/docker/pymc_env
python3.8 -m venv /mnt/docker/pymc_env
source /mnt/docker/pymc_env/bin/activate
pip3 install wheel
pip3 install -r /mnt/requirements_pymc.txt
deactivate
ln -s /mnt/docker/pymc_env/bin/activate /home/flow/activate_pymc

# tinyDA env
mkdir /mnt/docker/tinyda_env
python3.8 -m venv /mnt/docker/tinyda_env
source /mnt/docker/tinyda_env/bin/activate
pip3 install wheel
pip3 install -r /mnt/requirements_tinyda.txt
deactivate
ln -s /mnt/docker/tinyda_env/bin/activate /home/flow/activate_tinyda

cd /home/flow