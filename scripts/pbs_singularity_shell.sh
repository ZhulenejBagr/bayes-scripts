#!/bin/bash

#PBS -S /bin/bash
#PBS -N singularity_sample_script_1000
#PBS -q charon
##PBS -l select=1:ncpus=10:mem=10gb:scratch_local=5gb
#PBS -l walltime=48:00:00

# run fterm sing atleast once to create the image
#./bin/fterm_sing
cd $PBS_O_WORKDIR # potential cause of symlinks being overwritten

cfg_path=$CFG_PATH

id=$(echo "$PBS_JOBID" | cut -d'.' -f1)
echo $id

link="${HOME}"/"${id: -3}"
echo $link

if [ ! -f "${link}" ] ; then
    rm "${link}"
fi
ln -s $SCRATCHDIR $link

singularity exec bp_simunek.sif bash scripts/singularity_run_script.sh "${mnt}" "${cfg_path}"
