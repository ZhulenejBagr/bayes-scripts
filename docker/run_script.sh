#!/bin/bash

cd /home/flow
sed -i -e 's/\r$//' /mnt/docker/setup_env.sh #(pokud nejde zanout setup_env.sh)
/mnt/docker/setup_env.sh
# run pymc stuff
source activate_pymc
sed -i -e 's/\r$//' /mnt/scripts/generate_and_plot.sh
/mnt/scripts/generate_and_plot.sh