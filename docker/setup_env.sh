#!/bin/bash
python3.8 -m venv /home/flow/.virtualenvs/bruh
source /home/flow/.virtualenvs/bruh/bin/activate
pip3 install wheel
pip3 install -r /mnt/docker/requirements.txt
deactivate
ln -s /home/flow/.virtualenvs/bruh/bin/activate /home/flow/activate_venv