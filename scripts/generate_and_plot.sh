#!/bin/bash
# wrapper for idata genearator and idata plotter

cd /mnt/
#python3.8 /mnt/scripts/idata_generator.py
python3.8 -m scripts.idata_generator
#python3.8 /mnt/scripts/idata_plotter.py
python3.8 -m scripts.idata_plotter