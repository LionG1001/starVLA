#!/bin/bash

HOSTFILE=$1
CHECK_INTERVAL=${CHECK_INTERVAL:-10}
RUN_ID=$(date +%Y%m%d_%H%M%S)
MARKER="/home/blake/.copy_done_${RUN_ID}"

NUM_NODES=$(grep -v '^#\|^$' "$HOSTFILE" | wc -l)
echo "NUM_NODES: $NUM_NODES"

mapfile -t hostlist < <(grep -v '^#\|^$' "$HOSTFILE" | awk '{print $1}')
for host in "${hostlist[@]}"; do
    (
        echo "$host check lib."
        ssh "$host" "ls -l /usr/lib/x86_64-linux-gnu/libmusa.so.4"
    )
done
