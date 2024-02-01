#!/bin/bash
cd /
mkdir /mnt/docker/venv
python3.8 -m venv /mnt/docker/venv
source /mnt/docker/venv/bin/activate
pip3 install wheel
pip3 install -r /mnt/docker/requirements.txt
deactivate
ln -s /mnt/docker/venv/bin/activate /home/flow/activate_venv
cd /home/flow