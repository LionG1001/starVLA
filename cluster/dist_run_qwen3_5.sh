#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3.5 StarVLA training on every host in HOSTFILE.
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
BASE_VLM="${BASE_VLM:-/home/jd/gl_dev/models/Qwen3.5-4B}"
CONFIG_YAML="${CONFIG_YAML:-${WORKDIR}/examples/Robotwin/train_files/starvla_cotrain_robotwin_qwen35_abs.yaml}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/home/jd/blake/starvla/playground/Datasets/RoboTwin}"
DATA_MIX="${DATA_MIX:-robotwin_all_50}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${DATA_MIX}_qwen3_5_sdpa_math_${NNODES}n}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-150000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
# Native AdamW is the validated default for the current Qwen3.5/MUSA shape;
# keep FusedAdamW as an explicit opt-in for version-specific A/B testing.
STARVLA_ENABLE_FUSED_OPTIMIZER="${STARVLA_ENABLE_FUSED_OPTIMIZER:-0}"
STARVLA_ALLOW_TF32="${STARVLA_ALLOW_TF32:-auto}"
STARVLA_QWEN35_FLA_FASTPATH="${STARVLA_QWEN35_FLA_FASTPATH:-0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SDPA_BACKEND="${SDPA_BACKEND:-math}"
GPU_PEAK_TFLOPS="${GPU_PEAK_TFLOPS:-460.0}"
IS_RESUME="${IS_RESUME:-false}"
LOCAL_SIZE="${LOCAL_SIZE:-8}"

echo "Number of nodes: ${NNODES}"
echo "Master address: ${MASTER_ADDR}"
echo "Run ID: ${RUN_ID}"
echo "Attention backend: ${ATTN_IMPLEMENTATION}/${SDPA_BACKEND}"
echo "TF32 policy: ${STARVLA_ALLOW_TF32}"
echo "Experimental MUSA FLA fast path: ${STARVLA_QWEN35_FLA_FASTPATH}"
echo "Logs: ${LOG_DIR}"
echo "Checkpoints root: ${OUTPUT_DIR}"

build_preflight_command() {
  printf 'cd %q && test -f %q && test -d %q && test -f %q && test -d %q' \
    "${WORKDIR}" "${TRAIN_ENTRY}" "${BASE_VLM}" "${CONFIG_YAML}" "${DATA_ROOT_DIR}"
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
  printf 'cd %q && nohup env PYTHONUNBUFFERED=1 NNODES=%q NRANK=%q MASTER_ADDR=%q MAIN_PROCESS_PORT=%q OUTPUT_DIR=%q RUN_ID=%q BASE_VLM=%q CONFIG_YAML=%q DATA_ROOT_DIR=%q DATA_MIX=%q PER_DEVICE_BATCH_SIZE=%q MAX_TRAIN_STEPS=%q SAVE_INTERVAL=%q LOGGING_FREQUENCY=%q EVAL_INTERVAL=%q STARVLA_ENABLE_FUSED_OPTIMIZER=%q STARVLA_ALLOW_TF32=%q STARVLA_QWEN35_FLA_FASTPATH=%q WANDB_MODE=%q ATTN_IMPLEMENTATION=%q SDPA_BACKEND=%q GPU_PEAK_TFLOPS=%q IS_RESUME=%q LOCAL_SIZE=%q bash %q > %q 2>&1 < /dev/null &' \
    "${WORKDIR}" "${NNODES}" "${rank}" "${MASTER_ADDR}" "${MAIN_PROCESS_PORT}" \
    "${OUTPUT_DIR}" "${RUN_ID}" "${BASE_VLM}" "${CONFIG_YAML}" "${DATA_ROOT_DIR}" \
    "${DATA_MIX}" "${PER_DEVICE_BATCH_SIZE}" "${MAX_TRAIN_STEPS}" "${SAVE_INTERVAL}" \
    "${LOGGING_FREQUENCY}" "${EVAL_INTERVAL}" "${STARVLA_ENABLE_FUSED_OPTIMIZER}" \
    "${STARVLA_ALLOW_TF32}" "${STARVLA_QWEN35_FLA_FASTPATH}" "${WANDB_MODE}" \
    "${ATTN_IMPLEMENTATION}" "${SDPA_BACKEND}" "${GPU_PEAK_TFLOPS}" "${IS_RESUME}" \
    "${LOCAL_SIZE}" "${TRAIN_ENTRY}" "${log_file}"
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
