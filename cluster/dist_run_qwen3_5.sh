#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3.5 StarVLA training on every host in HOSTFILE.
# This file only owns distributed orchestration. Model and training settings
# belong to TRAIN_ENTRY and its YAML config, and are not injected here. Bounded
# profiler controls are runtime instrumentation and may be forwarded explicitly.
# Usage:
#   bash cluster/dist_run_qwen3_5.sh HOSTFILE \
#     [--logdir LOG_DIR] [--output-dir OUTPUT_DIR] [--dry-run]

HOSTFILE=""
LOG_DIR=""
OUTPUT_DIR=""
DRY_RUN=0
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
WORKDIR="${WORKDIR:-/home/jd/gl_dev/starVLA}"
TRAIN_ENTRY="examples/Robotwin/train_files/run_robotwin_train_qwen3_5_musa.sh"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-120}"
STARTUP_POLL_INTERVAL_SECONDS="${STARTUP_POLL_INTERVAL_SECONDS:-2}"
STARVLA_PROFILE_ENABLED="${STARVLA_PROFILE_ENABLED:-0}"
STARVLA_PROFILE_RANKS="${STARVLA_PROFILE_RANKS:-0}"
STARVLA_PROFILE_WAIT="${STARVLA_PROFILE_WAIT:-2}"
STARVLA_PROFILE_WARMUP="${STARVLA_PROFILE_WARMUP:-1}"
STARVLA_PROFILE_ACTIVE="${STARVLA_PROFILE_ACTIVE:-3}"
STARVLA_PROFILE_REPEAT="${STARVLA_PROFILE_REPEAT:-1}"
STARVLA_PROFILE_GZIP="${STARVLA_PROFILE_GZIP:-1}"
STARVLA_PROFILE_MEMORY="${STARVLA_PROFILE_MEMORY:-1}"
STARVLA_PROFILE_STACK="${STARVLA_PROFILE_STACK:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir)
      LOG_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 HOSTFILE [--logdir LOG_DIR] [--output-dir OUTPUT_DIR] [--dry-run]"
      exit 0
      ;;
    *)
      if [[ -z "${HOSTFILE}" ]]; then
        HOSTFILE="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${HOSTFILE}" || ! -f "${HOSTFILE}" ]]; then
  echo "Error: a readable HOSTFILE is required: ${HOSTFILE:-<empty>}" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/${CURRENT_TIME}_qwen3_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_DIR}}"

if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${SCRIPT_DIR}/${LOG_DIR}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
NNODES=${#HOSTS[@]}
if ((NNODES == 0)); then
  echo "Error: no hosts found in ${HOSTFILE}" >&2
  exit 1
fi
if [[ $(printf '%s\n' "${HOSTS[@]}" | sort -u | wc -l) -ne ${NNODES} ]]; then
  echo "Error: duplicate hosts found in ${HOSTFILE}" >&2
  exit 1
fi
if ! [[ "${STARTUP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: STARTUP_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
if ! [[ "${STARTUP_POLL_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: STARTUP_POLL_INTERVAL_SECONDS must be a positive integer" >&2
  exit 1
fi

MASTER_ADDR=${HOSTS[0]}
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen3_5_bs4_eager_${NNODES}n}"
LOCAL_SIZE="${LOCAL_SIZE:-8}"

echo "Number of nodes: ${NNODES}"
echo "Master address: ${MASTER_ADDR}"
echo "Run ID: ${RUN_ID}"
echo "Logs: ${LOG_DIR}"
echo "Checkpoints root: ${OUTPUT_DIR}"
echo "Profiler: enabled=${STARVLA_PROFILE_ENABLED}, ranks=${STARVLA_PROFILE_RANKS}, wait=${STARVLA_PROFILE_WAIT}, warmup=${STARVLA_PROFILE_WARMUP}, active=${STARVLA_PROFILE_ACTIVE}, repeat=${STARVLA_PROFILE_REPEAT}, memory=${STARVLA_PROFILE_MEMORY}, stack=${STARVLA_PROFILE_STACK}"

build_preflight_command() {
  printf 'cd %q && test -f %q' "${WORKDIR}" "${TRAIN_ENTRY}"
}

echo "Validating SSH access and shared paths on every node..."
PREFLIGHT_COMMAND=$(build_preflight_command)
for host in "${HOSTS[@]}"; do
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" "${PREFLIGHT_COMMAND}"; then
    echo "Error: preflight failed on ${host}" >&2
    exit 1
  fi
done

find_free_port_on_host() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" bash <<'REMOTE_EOF' 2>/dev/null || true
for port in $(seq 20000 30000); do
  if command -v ss >/dev/null 2>&1; then
    if ! ss -ltn 2>/dev/null | grep -q ":${port} "; then
      echo "${port}"
      exit 0
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if ! netstat -ltn 2>/dev/null | grep -q ":${port} "; then
      echo "${port}"
      exit 0
    fi
  elif ! (echo >/dev/tcp/127.0.0.1/${port}) 2>/dev/null; then
    echo "${port}"
    exit 0
  fi
done
REMOTE_EOF
}

if [[ -z "${MAIN_PROCESS_PORT}" ]]; then
  MAIN_PROCESS_PORT=$(find_free_port_on_host "${MASTER_ADDR}")
  if [[ -z "${MAIN_PROCESS_PORT}" ]]; then
    MAIN_PROCESS_PORT="${DEFAULT_MAIN_PROCESS_PORT:-58887}"
    echo "Warning: failed to detect a free port; using ${MAIN_PROCESS_PORT}" >&2
  fi
fi
echo "Main process port: ${MAIN_PROCESS_PORT}"

build_remote_command() {
  local rank="$1"
  local log_file="$2"
  printf 'cd %q && nohup env PYTHONUNBUFFERED=1 NNODES=%q NRANK=%q MASTER_ADDR=%q MAIN_PROCESS_PORT=%q OUTPUT_DIR=%q RUN_ID=%q LOCAL_SIZE=%q STARVLA_PROFILE_ENABLED=%q STARVLA_PROFILE_RANKS=%q STARVLA_PROFILE_WAIT=%q STARVLA_PROFILE_WARMUP=%q STARVLA_PROFILE_ACTIVE=%q STARVLA_PROFILE_REPEAT=%q STARVLA_PROFILE_GZIP=%q STARVLA_PROFILE_MEMORY=%q STARVLA_PROFILE_STACK=%q bash %q > %q 2>&1 < /dev/null &' \
    "${WORKDIR}" "${NNODES}" "${rank}" "${MASTER_ADDR}" "${MAIN_PROCESS_PORT}" \
    "${OUTPUT_DIR}" "${RUN_ID}" "${LOCAL_SIZE}" "${STARVLA_PROFILE_ENABLED}" \
    "${STARVLA_PROFILE_RANKS}" "${STARVLA_PROFILE_WAIT}" \
    "${STARVLA_PROFILE_WARMUP}" "${STARVLA_PROFILE_ACTIVE}" \
    "${STARVLA_PROFILE_REPEAT}" "${STARVLA_PROFILE_GZIP}" \
    "${STARVLA_PROFILE_MEMORY}" "${STARVLA_PROFILE_STACK}" \
    "${TRAIN_ENTRY}" "${log_file}"
}

if ((DRY_RUN == 1)); then
  for rank in "${!HOSTS[@]}"; do
    host=${HOSTS[$rank]}
    log_file="${LOG_DIR}/log.${rank}.${host}"
    echo "DRY-RUN rank=${rank} host=${host}"
    build_remote_command "${rank}" "${log_file}"
    echo
  done
  echo "Dry run complete; no training process was submitted."
  exit 0
fi

launch_failures=0
for rank in "${!HOSTS[@]}"; do
  host=${HOSTS[$rank]}
  log_file="${LOG_DIR}/log.${rank}.${host}"
  remote_command=$(build_remote_command "${rank}" "${log_file}")
  echo "Launching machine rank ${rank} on ${host}; log: ${log_file}"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -f -n "${host}" "${remote_command}"; then
    echo "Error: failed to submit machine rank ${rank} on ${host}" >&2
    launch_failures=$((launch_failures + 1))
  fi
done

if ((launch_failures > 0)); then
  echo "Error: ${launch_failures} node launch command(s) failed; inspect ${LOG_DIR}" >&2
  exit 1
fi

launcher_is_running() {
  local host="$1"
  local rank="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" \
    "ps -eo args= | grep -F 'accelerate.commands.accelerate_cli launch' | grep -F -- '--machine_rank ${rank}' | grep -F -- '--main_process_port ${MAIN_PROCESS_PORT}' | grep -v grep >/dev/null"
}

echo "All ${NNODES} launch commands were submitted; validating launchers..."
startup_deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
startup_ready=0
pending_launchers=()

while ((SECONDS < startup_deadline)); do
  pending_launchers=()
  for rank in "${!HOSTS[@]}"; do
    host=${HOSTS[$rank]}
    if ! launcher_is_running "${host}" "${rank}"; then
      pending_launchers+=("rank=${rank}@${host}")
    fi
  done

  if ((${#pending_launchers[@]} == 0)); then
    startup_ready=1
    break
  fi
  sleep "${STARTUP_POLL_INTERVAL_SECONDS}"
done

if ((startup_ready == 0)); then
  echo "Error: launcher validation timed out after ${STARTUP_TIMEOUT_SECONDS}s" >&2
  echo "Missing launchers: ${pending_launchers[*]}" >&2
  for rank in "${!HOSTS[@]}"; do
    host=${HOSTS[$rank]}
    log_file="${LOG_DIR}/log.${rank}.${host}"
    echo "===== tail ${log_file} =====" >&2
    if [[ -f "${log_file}" ]]; then
      tail -n 40 "${log_file}" >&2
    else
      echo "log file does not exist" >&2
    fi
  done
  exit 1
fi

echo "All ${NNODES} Qwen3.5 machine launchers are running. Logs: ${LOG_DIR}"
