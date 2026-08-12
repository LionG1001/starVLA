#!/bin/bash

export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MUSA_LAUNCH_BLOCKING=0
export MUSA_DEVICE_MAX_CONNECTIONS=1
export MUSA_EXECUTION_TIMEOUT=3200000
export MCCL_CROSS_NIC=0
export MCCL_SOCKET_IFNAME=bond0

ulimit -n 524288
ulimit -u 9286429
ulimit -s unlimited
ulimit -c unlimited

# =============================================================================
# MFU (Model FLOPs Utilization) Optimization Strategies for 500T GPU
# =============================================================================
# MFU = (Actual TFLOPs / Peak TFLOPs) * 100%
# Target: Achieve >50% MFU on MUSA GPU
#
# ===== 1. BATCH SIZE & SEQUENCE LENGTH OPTIMIZATION =====
# Current: per_device_batch_size=1, gradient_accumulation_steps (check config)
# Recommendation:
#   - Increase per_device_batch_size as much as possible before OOM
#   - Use gradient accumulation to maintain effective batch size
#   - For 500T GPU with ~80GB memory, try per_device_batch_size=4-8
#
# ===== 2. MIXED PRECISION TRAINING =====
# Current: torch.bfloat16 (already using)
# This reduces memory and increases throughput by ~2x
# Ensure MUSA supports bf16 for optimal performance
#
# ===== 3. GRADIENT ACCUMULATION STEPS =====
# Calculate: effective_batch_size = per_device_batch_size * num_gpus * grad_accum_steps
# For 8 GPUs with per_device_batch_size=4, grad_accum_steps=4: effective_batch_size=128
# Higher batch sizes = better GPU utilization and MFU
#
# ===== 4. DATA LOADING OPTIMIZATION =====
# Current: num_worker=0 (suboptimal)
# Recommendation: Set num_worker=4-8 to avoid CPU bottleneck
# Use pin_memory=True if available on MUSA
#
# ===== 5. COMPILATION & OPTIMIZATION FLAGS =====
# Consider using torch.compile() for PyTorch 2.0+
# export MUSA_GRAPH_CAPTURE=1 (if supported)
#
# ===== 6. COMMUNICATION OPTIMIZATION =====
# Already set: MCCL_BUFFSIZE, MCCL_ALGOS
# Consider: gradient compression for large models
#
# ===== 7. CHECKPOINT & IO OPTIMIZATION =====
# Use async checkpointing to avoid blocking
# Save to fast storage (NVMe/SSD)
#
# ===== RECOMMENDED CONFIGURATION FOR 500T GPU =====
# per_device_batch_size=4-8
# gradient_accumulation_steps=4-8
# num_worker=4-8
# Use bf16 mixed precision
# Enable torch.compile() if available
#
# ===== MONITORING MFU =====
# The training script now logs:
# - model_tflops_per_step: Theoretical model TFLOPs per step
# - throughput_samples_per_sec: Samples processed per second
# - mfu_percent: Model FLOPs Utilization percentage
#
# Target MFU: >50% for good utilization, >70% for excellent
###########################################################################################

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
workdir=/home/jd/blake/starvla
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
datadir=/home/qwen3vl
libero_data_root=$datadir/playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_goal
run_root_dir=./results/Checkpoints
run_id=jd_libero4in1_qwen3oft_musa
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

NNODES=${NNODES:-"1"}
NODE_RANK=${NRANK:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MAIN_PROCESS_PORT:-"58887"}
LOCAL_SIZE=8
NUM_PROCESS=$((NNODES * LOCAL_SIZE))
ulimit -n 524288
ulimit -s unlimited
ulimit -c unlimited
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
accelerate launch \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --num_machines ${NNODES} \
  --config_file $workdir/starVLA/config/deepseeds/deepspeed_zero1_musa.yaml \
  --num_processes $NUM_PROCESS \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 20 \
  --trainer.vla_data.video_backend decord \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 100 \
  --trainer.gpu_peak_tflops 480.0 \
  --num_worker 2 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}

  # 
  # --wandb_project starVLA_Libero \
  # --wandb_entity jinhuiye
