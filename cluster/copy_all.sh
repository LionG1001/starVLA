#!/bin/bash

HOSTFILE=$1
NUM_NODES=$(grep -v '^#\|^$' $HOSTFILE | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    (
        ssh -f -n "$host" "mkdir -p /home/blake"
        ssh -f -n "$host" "cp -r /home/jd/blake/mtwan_assets/relaion-art-train /home/blake/"
        echo "$host is copy."
    ) &
done

wait
