export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=900000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MUSA_LAUNCH_BLOCKING=0
export MUSA_DEVICE_MAX_CONNECTIONS=1
export MUSA_EXECUTION_TIMEOUT=900000
export MCCL_CROSS_NIC=0
export MCCL_SOCKET_IFNAME=bond0

ulimit -n 524288
ulimit -u 9286429
ulimit -s unlimited
ulimit -c unlimited

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin_all_50
run_id=0424_${data_mix}_qwen3OFT_all
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
export DS_ACCELERATOR=musa

accelerate launch \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --num_machines ${NNODES} \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_musa.yaml \
  --num_processes $NUM_PROCESS \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 6 \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 16000000 \
  --trainer.save_interval 1500 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 1000 \
  --trainer.is_resume \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project sta10000 rVLA_Robotwin \
  4-wandb_entity axi-the-cat \
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

