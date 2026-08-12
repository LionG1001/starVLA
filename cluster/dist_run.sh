#!/bin/bash
# Launch StarVLA training on every host in HOSTFILE.
# export NCCL_IB_SL=1
# Usage:
#   bash dist_run.sh HOSTFILE [--logdir LOG_DIR] [--output-dir OUTPUT_DIR]
HOSTFILE=""
LOG_DIR=""
OUTPUT_DIR=""
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
WORKDIR="${WORKDIR:-/home/jd/gl_dev/starVLA}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-120}"
STARTUP_POLL_INTERVAL_SECONDS="${STARTUP_POLL_INTERVAL_SECONDS:-2}"
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

if ((NNODES == 0)); then
  echo "Error: no hosts found in ${HOSTFILE}" >&2
  exit 1
fi

if ! [[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: STARTUP_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi

if ! [[ "$STARTUP_POLL_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: STARTUP_POLL_INTERVAL_SECONDS must be a positive integer" >&2
  exit 1
fi

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
    MAIN_PROCESS_PORT="${DEFAULT_MAIN_PROCESS_PORT:-58887}"
    echo "Warning: failed to detect free port on ${MASTER_ADDR}, using ${MAIN_PROCESS_PORT}"
  fi
fi

launch_failures=0
for host in "${ip_list[@]}"; do
  rank=$COUNT
  log_file="$LOG_DIR/log.$rank.$host"
  echo "Launching machine rank ${rank} on ${host}; log: ${log_file}"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -f -n "$host" \
    "bash -c 'cd \"$WORKDIR\" && PYTHONUNBUFFERED=1 NNODES=$NNODES NRANK=$rank MASTER_ADDR=$MASTER_ADDR MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT OUTPUT_DIR=\"$OUTPUT_DIR\" bash examples/Robotwin/train_files/run_robotwin_train_musa.sh > \"$log_file\" 2>&1 &'"; then
    echo "Error: failed to submit machine rank ${rank} on ${host}" >&2
    launch_failures=$((launch_failures + 1))
  fi
  # 1 host
  # ssh -f -n $host "bash -c 'cd $WORKDIR; NNODES=1 NRANK=0 OUTPUT_DIR=$OUTPUT_DIR bash examples/LIBERO/train_files/run_libero_train_musa.sh > $LOG_DIR/log.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd $WORKDIR; NNODES=1 NRANK=0 OUTPUT_DIR=$OUTPUT_DIR bash examples/Robotwin/train_files/run_robotwin_train_musa.sh > $LOG_DIR/log.$COUNT.$host 2>&1 &'"
  COUNT=$((COUNT + 1))
done

if ((launch_failures > 0)); then
  echo "Error: ${launch_failures} node launch command(s) failed; inspect ${LOG_DIR}" >&2
  exit 1
fi

launcher_is_running() {
  local host="$1"
  local rank="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "ps -eo args= | grep -F 'accelerate.commands.accelerate_cli launch' | grep -F -- '--machine_rank ${rank}' | grep -F -- '--main_process_port ${MAIN_PROCESS_PORT}' | grep -v grep >/dev/null"
}

echo "All ${NNODES} node launch commands were submitted; validating launchers..."
startup_deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
startup_ready=0
pending_launchers=()

while ((SECONDS < startup_deadline)); do
  pending_launchers=()
  for rank in "${!ip_list[@]}"; do
    host=${ip_list[$rank]}
    if ! launcher_is_running "$host" "$rank"; then
      pending_launchers+=("rank=${rank}@${host}")
    fi
  done

  if ((${#pending_launchers[@]} == 0)); then
    startup_ready=1
    break
  fi

  sleep "$STARTUP_POLL_INTERVAL_SECONDS"
done

if ((startup_ready == 0)); then
  echo "Error: launcher validation timed out after ${STARTUP_TIMEOUT_SECONDS}s" >&2
  echo "Missing launchers: ${pending_launchers[*]}" >&2
  for rank in "${!ip_list[@]}"; do
    host=${ip_list[$rank]}
    log_file="$LOG_DIR/log.$rank.$host"
    echo "===== tail ${log_file} =====" >&2
    if [[ -f "$log_file" ]]; then
      tail -n 40 "$log_file" >&2
    else
      echo "log file does not exist" >&2
    fi
  done
  exit 1
fi

echo "All ${NNODES} machine launchers are running. Logs: ${LOG_DIR}"
