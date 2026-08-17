#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_KERNEL_TIMEOUT="${MUSA_KERNEL_TIMEOUT:-3200000}"
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MUSA_BLOCK_SCHEDULE_MODE="${MUSA_BLOCK_SCHEDULE_MODE:-1}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export MCCL_NET_SHARED_BUFFERS="${MCCL_NET_SHARED_BUFFERS:-0}"
export MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING:-0}"
export MUSA_DEVICE_MAX_CONNECTIONS="${MUSA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export MCCL_CROSS_NIC="${MCCL_CROSS_NIC:-0}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-bond0}"
export STARVLA_ENABLE_FUSED_OPTIMIZER="${STARVLA_ENABLE_FUSED_OPTIMIZER:-1}"
export STARVLA_PYAV_THREADS="${STARVLA_PYAV_THREADS:-1}"
# auto preserves the framework defaults. Set 0 for the Qwen3.5 RoPE TF32
# isolation run so every Accelerate/DeepSpeed rank uses full FP32 matmul.
export STARVLA_ALLOW_TF32="${STARVLA_ALLOW_TF32:-auto}"
# FLA is opt-in until its numerical and performance baselines are validated.
export STARVLA_QWEN35_FLA_FASTPATH="${STARVLA_QWEN35_FLA_FASTPATH:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Optional bounded profiling. It is disabled for normal training.
export STARVLA_PROFILE_ENABLED="${STARVLA_PROFILE_ENABLED:-0}"
export STARVLA_PROFILE_RANKS="${STARVLA_PROFILE_RANKS:-0}"
export STARVLA_PROFILE_WAIT="${STARVLA_PROFILE_WAIT:-2}"
export STARVLA_PROFILE_WARMUP="${STARVLA_PROFILE_WARMUP:-1}"
export STARVLA_PROFILE_ACTIVE="${STARVLA_PROFILE_ACTIVE:-3}"
export STARVLA_PROFILE_REPEAT="${STARVLA_PROFILE_REPEAT:-1}"
export STARVLA_PROFILE_GZIP="${STARVLA_PROFILE_GZIP:-1}"

ulimit -n 524288
ulimit -u 9286429
ulimit -s unlimited
ulimit -c unlimited

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenOFT}"
FREEZE_MODULE_LIST="${FREEZE_MODULE_LIST:-}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/examples/Robotwin/train_files/starvla_cotrain_robotwin_qwen35_abs.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${OUTPUT_DIR:-${REPO_ROOT}/results/Checkpoints}}"
DATA_MIX="${DATA_MIX:-robotwin_all_50}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${DATA_MIX}_qwen3_5_sdpa_math}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SDPA_BACKEND="${SDPA_BACKEND:-math}"
GPU_PEAK_TFLOPS="${GPU_PEAK_TFLOPS:-460.0}"

select_existing_directory() {
  local description="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "Error: could not locate ${description}; checked: $*" >&2
  return 1
}

# Explicit environment variables always win. The candidates keep the same
# entry point usable in a local checkout, the K8s workspace, and the internal
# gl-vla container without editing the YAML for every environment.
if [[ -z "${BASE_VLM:-}" ]]; then
  BASE_VLM=$(select_existing_directory "Qwen3.5 model directory" \
    "${REPO_ROOT}/models/Qwen3.5-4B" \
    "/home/jd/gl_dev/models/Qwen3.5-4B" \
    "/data/share/liang.geng/vla-project/models/Qwen3.5-4B")
fi
if [[ -z "${DATA_ROOT_DIR:-}" ]]; then
  DATA_ROOT_DIR=$(select_existing_directory "RoboTwin dataset directory" \
    "${REPO_ROOT}/playground/Datasets/RoboTwin" \
    "/home/jd/blake/starvla/playground/Datasets/RoboTwin" \
    "/data/share/liang.geng/vla-project/playground/Datasets/RoboTwin")
fi

if [[ ! -d "${BASE_VLM}" ]]; then
  echo "Error: Qwen3.5 model directory does not exist: ${BASE_VLM}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Error: training config does not exist: ${CONFIG_YAML}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT_DIR}" ]]; then
  echo "Error: RoboTwin dataset directory does not exist: ${DATA_ROOT_DIR}" >&2
  exit 1
fi
if [[ "${ATTN_IMPLEMENTATION}" != "sdpa" || "${SDPA_BACKEND}" != "math" ]]; then
  echo "Warning: this entry is validated with SDPA math, but received ${ATTN_IMPLEMENTATION}/${SDPA_BACKEND}." >&2
fi

OUTPUT_PATH="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_PATH}/"
export STARVLA_PROFILE_DIR="${STARVLA_PROFILE_DIR:-${OUTPUT_PATH}/traces}"

NNODES="${NNODES:-1}"
NODE_RANK="${NRANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MAIN_PROCESS_PORT:-58887}"
LOCAL_SIZE="${LOCAL_SIZE:-8}"
NUM_PROCESSES=$((NNODES * LOCAL_SIZE))
export DS_ACCELERATOR=musa

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]] \
  && "${REPO_ROOT}/.venv/bin/python" -c 'import accelerate, torch' >/dev/null 2>&1; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN=$(command -v python)
fi

case "${STARVLA_QWEN35_FLA_FASTPATH,,}" in
  1|true|yes|on)
    if ! "${PYTHON_BIN}" -c 'import fla' >/dev/null 2>&1; then
      echo "Error: STARVLA_QWEN35_FLA_FASTPATH=1 requires fla-core and flash-linear-attention." >&2
      echo "Install the validated 0.4.2 packages without replacing the MUSA torch/Triton stack." >&2
      exit 1
    fi
    ;;
esac

echo "Qwen3.5 training: nodes=${NNODES}, node_rank=${NODE_RANK}, processes=${NUM_PROCESSES}"
echo "Attention backend: ${ATTN_IMPLEMENTATION}/${SDPA_BACKEND}"
echo "TF32 policy: ${STARVLA_ALLOW_TF32}"
echo "Experimental MUSA FLA fast path: ${STARVLA_QWEN35_FLA_FASTPATH}"
echo "Model directory: ${BASE_VLM}"
echo "Dataset directory: ${DATA_ROOT_DIR}"
echo "MFU BF16 peak assumption per device: ${GPU_PEAK_TFLOPS} TFLOPS"
echo "Run output: ${OUTPUT_PATH}"

"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --num_machines "${NNODES}" \
  --config_file starVLA/config/deepseeds/deepspeed_zero1_musa.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}" \
  --framework.qwenvl.sdpa_backend "${SDPA_BACKEND}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --trainer.freeze_modules "${FREEZE_MODULE_LIST}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-150000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-10000}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-1}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-1000}" \
  --trainer.gpu_peak_tflops "${GPU_PEAK_TFLOPS}" \
  --trainer.optimizer.fused "${STARVLA_ENABLE_FUSED_OPTIMIZER}" \
  --trainer.is_resume "${IS_RESUME:-false}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
