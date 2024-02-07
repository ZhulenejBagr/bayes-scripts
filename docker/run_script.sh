#!/bin/bash

cd /home/flow
sed -i -e 's/\r$//' /mnt/docker/setup_env.sh #(pokud nejde zanout setup_env.sh)
/mnt/docker/setup_env.sh
source activate_venv
sed -i -e 's/\r$//' #/mnt/scripts/generate_and_plot.sh
/mnt/scripts/generate_and_plot.sh