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
# Variable-length batches can fragment reserved memory. Expand allocator
# segments instead of retaining many unusable fragments between steps.
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export MCCL_CROSS_NIC="${MCCL_CROSS_NIC:-0}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-bond0}"
# On the validated MTT S5000/torch_musa stack, the native AdamW path is
# faster for this Qwen3.5 shape. Set STARVLA_ENABLE_FUSED_OPTIMIZER=1 only
# for an explicit A/B or on a stack where FusedAdamW has been revalidated.
export STARVLA_ENABLE_FUSED_OPTIMIZER="${STARVLA_ENABLE_FUSED_OPTIMIZER:-0}"
export STARVLA_PYAV_THREADS="${STARVLA_PYAV_THREADS:-1}"
# auto preserves the framework defaults. Set 0 for the Qwen3.5 RoPE TF32
# isolation run so every Accelerate/DeepSpeed rank uses full FP32 matmul.
export STARVLA_ALLOW_TF32="${STARVLA_ALLOW_TF32:-0}"
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
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${DATA_MIX}_qwen3_5_bs4_eager}"
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

OUTPUT_PATH="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_PATH}/"
cp "${CONFIG_YAML}" "${OUTPUT_PATH}/resolved_training_config.yaml"
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

# Attention selection has a single source of truth: CONFIG_YAML. Do not accept
# ATTN_IMPLEMENTATION/SDPA_BACKEND environment variables or CLI overrides here.
# This baseline launcher fails early if the selected YAML is no longer eager.
QWEN_CONFIG_SUMMARY=$(
  "${PYTHON_BIN}" - "${CONFIG_YAML}" <<'PY'
import sys

import yaml

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

qwenvl = config.get("framework", {}).get("qwenvl", {})
trainer = config.get("trainer", {})
attn_implementation = qwenvl.get("attn_implementation")
sdpa_backend = qwenvl.get("sdpa_backend", "auto")
if attn_implementation != "eager":
    raise SystemExit(
        "Error: the Qwen3.5 baseline launcher requires "
        f"framework.qwenvl.attn_implementation=eager in {config_path}; "
        f"found {attn_implementation!r}."
    )
fastpath_switches = (
    "musa_fla_fastpath",
    "musa_vision_patch_linear_fastpath",
    "musa_vision_flash_attention",
)


def config_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise SystemExit(f"Error: fast-path switch must be boolean, got {value!r}.")


switch_summary = " ".join(
    f"{name}={str(config_bool(qwenvl.get(name, False))).lower()}"
    for name in fastpath_switches
)
zero1_native_avg = config_bool(trainer.get("musa_zero1_native_avg", False))
if config_bool(qwenvl.get("musa_fla_fastpath", False)):
    try:
        import fla  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Error: framework.qwenvl.musa_fla_fastpath=true requires "
            "fla-core and flash-linear-attention."
        ) from error
print(
    f"attention={attn_implementation}/{sdpa_backend} {switch_summary} "
    f"musa_zero1_native_avg={str(zero1_native_avg).lower()}"
)
PY
)

echo "Qwen3.5 training: nodes=${NNODES}, node_rank=${NODE_RANK}, processes=${NUM_PROCESSES}"
echo "Qwen3.5 config from YAML: ${QWEN_CONFIG_SUMMARY}"
echo "TF32 policy: ${STARVLA_ALLOW_TF32}"
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
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
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
