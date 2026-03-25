# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

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
libero_cosmos_dataset_train = L(Dataset)(
    dataset_dir="datasets/libero_cosmos_mp4/train",
    num_frames=93,
    video_size=(256, 256),
    num_workers=8,
)
libero_cosmos_dataset_val = L(Dataset)(
    dataset_dir="datasets/libero_cosmos_mp4/val",
    num_frames=93,
    video_size=(256, 256),
)

# Effective batch size = 128 (mimic-video paper).
# With 8 GPUs and context_parallel_size=2, data-parallel ranks = 4.
# Per-GPU batch size = 128 / 4 = 32. Adjust if using fewer GPUs.
dataloader_train_libero_cosmos = L(DataLoader)(
    dataset=libero_cosmos_dataset_train,
    sampler=L(get_sampler)(dataset=libero_cosmos_dataset_train),
    batch_size=32,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_libero_cosmos = L(DataLoader)(
    dataset=libero_cosmos_dataset_val,
    sampler=L(get_sampler)(dataset=libero_cosmos_dataset_val),
    batch_size=32,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
#   --config=cosmos_predict2/configs/base/config.py -- \
#   experiment=predict2_video2world_training_2b_libero_cosmos
#
# Hyperparams from mimic-video paper (Table IV):
#   - Video backbone finetuning: 7k steps, lr=1.778e-4, batch=128, warmup=1000 steps
#   - Model: Cosmos-Predict2 2B, 480p, 10fps single-view
predict2_video2world_training_2b_libero_cosmos = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_10fps"},  # 480p 10fps variant
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "constant"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_libero_cosmos",
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
        context_parallel_size=2,
    ),
    dataloader_train=dataloader_train_libero_cosmos,
    dataloader_val=dataloader_val_libero_cosmos,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=7_000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=1.778e-4,
        weight_decay=0,
    ),
)

for _item in [
    predict2_video2world_training_2b_libero_cosmos,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
