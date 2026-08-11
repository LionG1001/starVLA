#!/bin/bash
# This example will start serving the Wan2.1 T2I model FSDP2 training
# export NCCL_IB_SL=1
# Usage:
#   bash dist_run_fsdp.sh HOSTFILE [--logdir LOG_DIR]
HOSTFILE=""
LOG_DIR=""
OUTPUT_DIR=""
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
COUNT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir) LOG_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 HOSTFILE [--logdir LOG_DIR] [--output-dir OUTPUT_DIR]"
      exit 0
      ;;
    *)
      if [[ -z "$HOSTFILE" ]]; then
        HOSTFILE="$1"
        shift
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$HOSTFILE" ]]; then
  echo "Error: HOSTFILE is required"
  echo "Usage: $0 HOSTFILE [--logdir LOG_DIR] [--output-dir OUTPUT_DIR]"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "$LOG_DIR" ]]; then
  # 使用下划线代替冒号，避免远程执行时路径解析问题
  CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
  LOG_DIR="${SCRIPT_DIR}/${CURRENT_TIME}"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$LOG_DIR"
fi

# 确保 LOG_DIR 是绝对路径
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="${SCRIPT_DIR}/${LOG_DIR}"
fi

# 确保 OUTPUT_DIR 是绝对路径
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
hostlist=$(grep -v '^#\|^$' "$HOSTFILE" | awk '{print $1}' | xargs)
read -ra ip_list <<< "$hostlist"
NNODES=${#ip_list[@]}

MASTER_ADDR=${ip_list[0]}

echo "number of nodes: ${NNODES}"
echo "master address: ${MASTER_ADDR}"

find_free_port_on_host() {
  local host="$1"
  local start=20000
  local end=30000
  # 使用 heredoc 避免复杂的转义问题
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash <<'REMOTE_EOF' 2>/dev/null || true
for p in $(seq 20000 30000); do
  if command -v ss >/dev/null 2>&1; then
    if ! ss -ltn 2>/dev/null | grep -q ":${p} "; then
      echo "$p"
      exit 0
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if ! netstat -ltn 2>/dev/null | grep -q ":${p} "; then
      echo "$p"
      exit 0
    fi
  else
    if ! (echo >/dev/tcp/127.0.0.1/${p}) 2>/dev/null; then
      echo "$p"
      exit 0
    fi
  fi
done
REMOTE_EOF
}


if [[ -z "$MAIN_PROCESS_PORT" ]]; then
  MAIN_PROCESS_PORT=$(find_free_port_on_host "$MASTER_ADDR")
  if [[ -n "$MAIN_PROCESS_PORT" ]]; then
    echo "Selected free main process port on ${MASTER_ADDR}: ${MAIN_PROCESS_PORT}"
  else
    echo "Warning: failed to detect free port on ${MASTER_ADDR}, using default"
  fi
fi

for host in ${ip_list[@]}; do
  echo $host
  ssh -f -n $host "bash -c 'cd /home/jd/blake/starvla; NNODES=$NNODES NRANK=$COUNT MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MAIN_PROCESS_PORT OUTPUT_DIR=$OUTPUT_DIR bash examples/Robotwin/train_files/run_robotwin_train_musa.sh > $LOG_DIR/log.$COUNT.$host 2>&1 &'"
  # 1 host
  # ssh -f -n $host "bash -c 'cd /home/jd/blake/starvla; NNODES=1 NRANK=0 OUTPUT_DIR=$OUTPUT_DIR bash examples/LIBERO/train_files/run_libero_train_musa.sh > $LOG_DIR/log.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd /home/jd/blake/starvla; NNODES=1 NRANK=0 OUTPUT_DIR=$OUTPUT_DIR bash examples/Robotwin/train_files/run_robotwin_train_musa.sh > $LOG_DIR/log.$COUNT.$host 2>&1 &'"
  if [ "$host" == "$MASTER_ADDR" ]; then
    LOG_FILE="$LOG_DIR/log.$COUNT.$host"
    echo "Waiting for master node log output..."
    while [ ! -s "$LOG_FILE" ]; do
      sleep 1
    done
    echo "Master node log has output, waiting 3 seconds..."
    sleep 3
  fi
  ((COUNT++))
done
