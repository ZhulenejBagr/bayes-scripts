#!/bin/bash

#PBS -S /bin/bash
#PBS -N singularity_sample_script_500
#PBS -q charon
#PBS -l select=1:ncpus=20:mem=20gb 
#PBS -l walltime=24:00:00

# run fterm sing atleast once to create the image
#./bin/fterm_sing

cd $PBS_O_WORKDIR

ln -s $SCRATCHDIR scratch

singularity exec bp_simunek.sif bash scripts/singularity_run_script.sh