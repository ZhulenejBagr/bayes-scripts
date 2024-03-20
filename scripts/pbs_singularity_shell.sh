#!/bin/bash

#PBS -S /bin/bash
#PBS -N singularity_sample_script_1000
#PBS -q charon
#PBS -l select=1:ncpus=1:mem=2gb 
#PBS -l walltime=32:00:00

# run fterm sing atleast once to create the image
#./bin/fterm_sing

cd $PBS_O_WORKDIR

singularity exec bp_simunek.sif bash scripts/singularity_run_script.sh