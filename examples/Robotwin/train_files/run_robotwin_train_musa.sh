#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_KERNEL_TIMEOUT="${MUSA_KERNEL_TIMEOUT:-3200000}"
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MUSA_LAUNCH_BLOCKING=0
export MUSA_DEVICE_MAX_CONNECTIONS="${MUSA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export MCCL_CROSS_NIC=0
export MCCL_SOCKET_IFNAME=bond0
export STARVLA_ENABLE_FUSED_OPTIMIZER="${STARVLA_ENABLE_FUSED_OPTIMIZER:-1}"
export STARVLA_PYAV_THREADS="${STARVLA_PYAV_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Bounded profiling is disabled by default. When enabled, rank 0 records a
# short MUSA trace after warmup and training then continues without profiling.
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

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm="${BASE_VLM:-/home/jd/blake/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
config_yaml="${CONFIG_YAML:-${REPO_ROOT}/examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml}"
run_root_dir="${RUN_ROOT_DIR:-${OUTPUT_DIR:-${REPO_ROOT}/results/Checkpoints}}"
data_mix="${DATA_MIX:-robotwin_all_50}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${data_mix}_qwen3OFT_all}"
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp "${BASH_SOURCE[0]}" "${output_dir}/"
export STARVLA_PROFILE_DIR="${STARVLA_PROFILE_DIR:-${output_dir}/traces}"

NNODES=${NNODES:-"1"}
NODE_RANK=${NRANK:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MAIN_PROCESS_PORT:-"58887"}
LOCAL_SIZE="${LOCAL_SIZE:-8}"
NUM_PROCESS=$((NNODES * LOCAL_SIZE))
export DS_ACCELERATOR=musa

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]] \
  && "${REPO_ROOT}/.venv/bin/python" -c 'import accelerate, torch' >/dev/null 2>&1; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN=$(command -v python)
fi

"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --num_machines ${NNODES} \
  --config_file starVLA/config/deepseeds/deepspeed_zero1_musa.yaml \
  --num_processes $NUM_PROCESS \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR:-/home/jd/blake/starvla/playground/Datasets/RoboTwin}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-4}" \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-150000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-10000}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-1}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-1000}" \
  --trainer.optimizer.fused "${STARVLA_ENABLE_FUSED_OPTIMIZER}" \
  --trainer.is_resume "${IS_RESUME:-false}" \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
  # --is_debug True

#  --trainer.max_train_steps 150000 \
##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
