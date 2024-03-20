#!/bin/bash

#PBS -S /bin/bash
#PBS -N interactive_singularity_shell
#PBS -q charon
#PBS -l select=1:ncpus=1:mem=2gb 
#PBS -l walltime=32:00:00

# run fterm sing atleast once to create the image
#./bin/fterm_sing

singularity exec bp_simunek.sif ./scripts/singularity_run_script.sh