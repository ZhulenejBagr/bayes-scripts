#!/bin/bash

#PBS -I
#PBS -S /bin/bash
#PBS -N interactive_singularity_shell
#PBS -q charon
#PBS -l select=1:ncpus=20:mem=5gb 
#PBS -l walltime=02:00:00

./bin/fterm_sing
