#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.env.secrets"
source "${REPO_ROOT}/.env.paths"

DATASET_PATH="${DATASET_PATH:-/nvme/datasets/teleop/preprocessed}"
NUM_FRAMES="${NUM_FRAMES:-61}"
LATENT_FRAMES="${LATENT_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-3}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nvme/checkpoints}"
TRAIN_WORKERS="${TRAIN_WORKERS:-32}"
VAL_WORKERS="${VAL_WORKERS:-8}"

export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"
export WANDB_ENTITY="${WANDB_ENTITY:-eth-robotics-club}"
export WANDB_PROJECT="${WANDB_PROJECT:-cosmos2-video}"

cd "${REPO_ROOT}"

# REMEMBER TO *CENTER CROP* VIDEO ON INFERENCE TO EXACT SPEC AS HERE

# currently doing either 1 or 5 frames conditioning
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  model.config.pipe_config.net.sac_config.mode=block_wise \
  model.config.pipe_config.net.sac_config.every_n_blocks=1 \
  experiment=predict2_video2world_lora_training_2b_cosmos_nemo_assets \
  model=predict2_video2world_fsdp_2b_480p_10fps \
  model.config.pipe_config.ema.enabled=False \
  dataloader_train.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_train.sampler.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_val.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_val.sampler.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_train.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_train.sampler.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_val.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_val.sampler.dataset.num_frames="${NUM_FRAMES}" \
  model.config.pipe_config.state_t="${LATENT_FRAMES}" \
  dataloader_train.batch_size="${BATCH_SIZE}" \
  dataloader_val.batch_size=1 \
  dataloader_train.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_train.sampler.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_val.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_val.sampler.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  model.config.train_architecture=lora \
  model.config.pipe_config.ema.enabled=False \
  model_parallel.context_parallel_size=1 \
  dataloader_train.num_workers="${TRAIN_WORKERS}" \
  trainer.run_validation=False \
  trainer.callbacks.draw_sample.is_ema=False \
  trainer.callbacks.draw_sample.run_at_start=True \
  trainer.logging_iter=10
  # dataloader_val.num_workers="${VAL_WORKERS}" \

# Params copied from cosmos-predict2.5/commands/train.sh. Uncomment and adapt as needed.
#
# experiment=predict2_video2world_training_2b_groot_gr1_480
# dataloader_train.dataset.num_frames=45
# dataloader_train.sampler.dataset.num_frames=45
# dataloader_train.dataset.dataset_dir=${NVME}/datasets/teleop/preprocessed
# dataloader_train.sampler.dataset.dataset_dir=${NVME}/datasets/teleop/preprocessed
# dataloader_train.dataloaders.video_data.dataloader.batch_size=128
# trainer.logging_iter=10
# trainer.callbacks.every_n_sample_reg.run_at_start=True
# trainer.callbacks.every_n_sample_reg.guidance=[3.0]
# trainer.callbacks.every_n_sample_reg.every_n=100
# trainer.callbacks.every_n_sample_ema.every_n=999999999
# model.config.use_lora=True 
# model.config.lora_rank=32
# model.config.lora_alpha=32
# "model.config.lora_target_modules='q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2'"
# model.config.init_lora_weights=True
# job.project=cosmos2.5-video
# job.group=2b_groot_gr1_480_run1
# optimizer.lr=0.0001726336
# model.config.min_num_conditional_frames=1
# model.config.max_num_conditional_frames=5
# job.name=push_box_lora-3
#
# dataloader_val.dataloaders.video_data.dataloader.batch_size=32
# dataloader_val.dataloaders.video_data.dataset.num_frames=45
