#!/bin/bash

#PBS -S /bin/bash
#PBS -N singularity_sample_script_1000
#PBS -q charon
#PBS -l select=1:ncpus=10:mem=10gb:scratch_local=5gb
#PBS -l walltime=36:00:00

# run fterm sing atleast once to create the image
#./bin/fterm_sing
cd $PBS_O_WORKDIR

cfg_path=$1

id=$($PBS_JOBID | cut -d'.' -f1)
echo $id

mnt="${HOME}"/."${id}"
echo $mnt

mount --bind $SCRATCHDIR $mnt

singularity exec bp_simunek.sif bash scripts/singularity_run_script.sh "${mnt}" "${cfg_path}"