#!/bin/bash

# get ip addreses of all nodes
NODES=`cat $PBS_NODEFILE`

# ip of head node
head_ip=`hostname -i`

# port to use
port=6379

# combine ip and port
head_address="$head_ip:$port"
echo $head_address

# change into workdir
cd $PBS_O_WORKDIR

# create container instance
echo "Starting container on head node"
singularity instance start bp_simunek.sif cont

echo "Running head node script"
# run head script inside singularity container
singularity exec instance://cont scripts/head_node_script.sh $head_address $port ${NODES[@]} $SSHPASS

