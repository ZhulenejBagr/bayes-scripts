#!/bin/bash

# get head address from input
head_address=$1

# get port from input
port=$2

# get worker node addresses from input
NODES=$3

# get pass to use with ssh
pass=$4

# get a copy of $PBS_O_WORKDIR, other nodes seem to not know its value
workdir=$(echo $PBS_O_WORKDIR)
echo "workdir $workdir"

# activate venv
source venv/bin/activate

# start ray on head node
echo "Starting Ray on head node"
ray start --head --port=$port --temp-dir=$SCRATCHDIR
sleep 5

# install sshpass, later wont be neccessary
echo apt-get install sshpass


# run node script on every node
echo "Starting worker nodes"
for NODE in $NODES; do
	command="cd $workdir; singularity instance start bp_simunek.sif cont; singularity exec instance://cont scripts/worker_node_script.sh $head_address $SCRATCHDIR; exit;"
	echo $command
	sshpass -e ssh -o StrictHostKeyChecking=no "${NODE}" "${command}"
	echo $?
done
