# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_predict2.callbacks.grad_clip import GradClip
from cosmos_predict2.callbacks.wandb_setup import WandbSetup
from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# num_frames=93: aligned with agibot config (state_t=24, temporal_compression_factor=4)
# formula: (state_t - 1) * temporal_compression + 1 = (24-1)*4+1 = 93
yams_dataset_train = L(Dataset)(
    dataset_dir="datasets/yams_cosmos_mp4/train",
    num_frames=93,
    video_size=(480, 640),
)
yams_dataset_val = L(Dataset)(
    dataset_dir="datasets/yams_cosmos_mp4/val",
    num_frames=93,
    video_size=(480, 640),
)

# batch_size=4 per GPU × 8 DP ranks (CP=1) × grad_accum=1 = effective batch 32
dataloader_train_yams = L(DataLoader)(
    dataset=yams_dataset_train,
    sampler=L(get_sampler)(dataset=yams_dataset_train),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_yams = L(DataLoader)(
    dataset=yams_dataset_val,
    sampler=L(get_sampler)(dataset=yams_dataset_val),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
#   --config=cosmos_predict2/configs/base/config.py -- \
#   experiment=predict2_video2world_training_2b_yams
#
# Compared to libero_cosmos:
#   video_size:            (256,256) → (480,640)   match dataset native resolution
#   batch_size (per GPU):  8         → 4            effective batch 128 → 32
#   context_parallel_size: 2         → 1            larger res needs no CP at eff. batch 32
#   grad_accum_iter:       2         → 1            not needed at this batch size
#   weight_decay:          0         → 0.1          regularisation for small dataset
#   grad_clip:             1.0       → 10.0         looser clip, dataset is small
#   scheduler:             constant  → lambdalinear  warmup 1000 steps then constant (f_min=1.0)
#   warmup_steps:          0         → 1000         mimic-video paper recommendation
#   lr decay:              none      → none          same as libero, flat after warmup
#   lr:                    1.778e-4  → 1.778e-4     kept; warmup handles the ramp-up
_VIZ_EVERY_N = 200

predict2_video2world_training_2b_yams = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_10fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "constant"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_yams",
    ),
    model=dict(
        config=dict(
            train_architecture="lora",
            lora_rank=16,
            lora_alpha=16,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,  # 8 GPUs → 8 DP ranks, effective batch = 4 × 8 = 32
    ),
    dataloader_train=dataloader_train_yams,
    dataloader_val=dataloader_val_yams,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
            wandb_setup=L(WandbSetup)(),
            grad_clip=L(GradClip)(clip_norm=10.0),  # 1.0 in libero
            draw_sample=L(EveryNDrawSample)(
                every_n=_VIZ_EVERY_N,
                is_x0=True,
                is_sample=False,  # skip expensive 35-step sampling
                is_ema=True,
                n_x0_level=2,
                n_viz_sample=1,
                show_all_frames=False,
                fps=10,
            ),
        ),
        max_iter=7_000,
        run_validation=True,
        validation_iter=_VIZ_EVERY_N,
        max_val_iter=10,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=1.778e-4,
        weight_decay=0.1,  # 0 in libero
    ),
)

for _item in [
    predict2_video2world_training_2b_yams,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
