#!/bin/bash

#PBS -S /bin/bash
#PBS -N interactive_singularity_shell
#PBS -q charon
#PBS -l select=1:ncpus=1:mem=2gb 
#PBS -l walltime=32:00:00

./bin/fterm_sing
source venv/bin/activate
python -m bp_simunek.tests.simulation.test_tinyda_simulation
