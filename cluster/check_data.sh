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
        ssh "$host" "mkdir -p /home/blake; rm -f '$MARKER'; cp -r /home/jd/blake/mtwan_assets/relaion-art-train /home/blake/; touch '$MARKER'"
        echo "$host copy finished."
    ) &
done

while true; do
    done_count=0
    for host in "${hostlist[@]}"; do
        if ssh "$host" "test -f '$MARKER'"; then
            done_count=$((done_count + 1))
        fi
    done
    echo "[$(date +%H:%M:%S)] copy done: $done_count/$NUM_NODES"
    if [ "$done_count" -ge "$NUM_NODES" ]; then
        break
    fi
    sleep "$CHECK_INTERVAL"
done

wait
