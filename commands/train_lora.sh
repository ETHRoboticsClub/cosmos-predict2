#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.env.secrets"
source "${REPO_ROOT}/.env.paths"

DATASET_PATH="${DATASET_PATH:-/nvme/datasets/teleop/preprocessed}"
NUM_FRAMES="${NUM_FRAMES:-45}"
LATENT_FRAMES="${LATENT_FRAMES:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nvme/checkpoints}"

TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
HELDOUT_VIDEO_INDEX="${HELDOUT_VIDEO_INDEX:-0}"
HELDOUT_START_INDICES="${HELDOUT_START_INDICES:-[0,16,32]}"
ENABLE_WANDB_VIDEO_SAMPLING="${ENABLE_WANDB_VIDEO_SAMPLING:-1}"
DRAW_SAMPLE_EVERY="${DRAW_SAMPLE_EVERY:-100}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-0}"
MAX_ITER="${MAX_ITER:-10000}"
SLOW_LOAD_WARN_S="${SLOW_LOAD_WARN_S:-2.0}"

# Disabled branch
DRAW_SAMPLE_ARGS=(
  trainer.callbacks.draw_sample.is_ema=False
  trainer.callbacks.draw_sample.is_x0=False
  trainer.callbacks.draw_sample.is_sample=False
  trainer.callbacks.draw_sample.run_at_start=False
  trainer.callbacks.draw_sample.every_n="${DRAW_SAMPLE_EVERY}"
)

# Enabled branch
if [[ "${ENABLE_WANDB_VIDEO_SAMPLING}" == "1" ]]; then
  DRAW_SAMPLE_ARGS=(
    trainer.callbacks.draw_sample.is_ema=False
    trainer.callbacks.draw_sample.is_x0=False
    trainer.callbacks.draw_sample.is_sample=True
    trainer.callbacks.draw_sample.show_all_frames=True
    trainer.callbacks.draw_sample.n_viz_sample=3
    trainer.callbacks.draw_sample.fixed_sample_video_index="${HELDOUT_VIDEO_INDEX}"
    trainer.callbacks.draw_sample.fixed_sample_start_indices="${HELDOUT_START_INDICES}"
    trainer.callbacks.draw_sample.fixed_sample_dataset_dir="${DATASET_PATH}"
    trainer.callbacks.draw_sample.fixed_sample_num_frames="${NUM_FRAMES}"
    trainer.callbacks.draw_sample.fixed_sample_video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
    trainer.callbacks.draw_sample.fps=10
    trainer.callbacks.draw_sample.run_at_start=True
    trainer.callbacks.draw_sample.every_n="${DRAW_SAMPLE_EVERY}"
  )
fi


export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"
export WANDB_ENTITY="${WANDB_ENTITY:-eth-robotics-club}"
export WANDB_PROJECT="${WANDB_PROJECT:-cosmos2-video}"
export COSMOS_SLOW_LOAD_WARN_S="${COSMOS_SLOW_LOAD_WARN_S:-${SLOW_LOAD_WARN_S}}"
if [[ "${DISABLE_TORCH_COMPILE}" == "1" ]]; then
  export TORCH_COMPILE_DISABLE=1
  export TORCHDYNAMO_DISABLE=1
fi

cd "${REPO_ROOT}"

# REMEMBER TO *CENTER CROP* VIDEO ON INFERENCE TO EXACT SPEC AS HERE

# currently doing either 1 or 5 frames conditioning
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_lora_training_2b_cosmos_nemo_assets \
  job.name=lora_200M \
  model=predict2_video2world_fsdp_2b_480p_10fps \
  model_parallel.context_parallel_size=1 \
  model.config.train_architecture=lora \
  model.config.lora_rank=140 \
  model.config.lora_alpha=140 \
  model.config.pipe_config.ema.enabled=False \
  model.config.pipe_config.ema.enabled=False \
  model.config.pipe_config.state_t="${LATENT_FRAMES}" \
  model.config.pipe_config.net.sac_config.mode=block_wise \
  model.config.pipe_config.net.sac_config.every_n_blocks=1 \
  dataloader_train.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_train.sampler.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_train.dataset.exclude_video_indices="[${HELDOUT_VIDEO_INDEX}]" \
  dataloader_train.sampler.dataset.exclude_video_indices="[${HELDOUT_VIDEO_INDEX}]" \
  dataloader_train.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_train.sampler.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_train.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_train.sampler.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_train.batch_size="${BATCH_SIZE}" \
  dataloader_train.num_workers="${TRAIN_WORKERS}" \
  dataloader_val.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_val.sampler.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_val.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_val.sampler.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_val.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_val.sampler.dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]" \
  dataloader_val.batch_size=1 \
  trainer.run_validation=False \
  "${DRAW_SAMPLE_ARGS[@]}" \
  trainer.max_iter="${MAX_ITER}" \
  trainer.logging_iter=5 \
  optimizer.lr=0.0001726336 \
  scheduler.cycle_lengths="[${MAX_ITER}]"
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
