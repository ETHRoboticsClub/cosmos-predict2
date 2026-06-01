# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_teleop_raw import TeleopRawDataset
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

teleop_raw_dataset_train = L(TeleopRawDataset)(
    dataset_dir="~/code/mimic-video/data/teleop_raw",
    num_frames=93,
    video_size=(256, 256),
    camera_name="camera_top",
    split="train",
    val_fraction=0.1,
    split_seed=42,
)
teleop_raw_dataset_val = L(TeleopRawDataset)(
    dataset_dir="~/code/mimic-video/data/teleop_raw",
    num_frames=93,
    video_size=(256, 256),
    camera_name="camera_top",
    split="val",
    val_fraction=0.1,
    split_seed=42,
)

dataloader_train_teleop_raw = L(DataLoader)(
    dataset=teleop_raw_dataset_train,
    sampler=L(get_sampler)(dataset=teleop_raw_dataset_train),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_teleop_raw = L(DataLoader)(
    dataset=teleop_raw_dataset_val,
    sampler=L(get_sampler)(dataset=teleop_raw_dataset_val),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

predict2_video2world_training_2b_teleop_raw = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b_480p_10fps"},
        {"override /optimizer": "adamw"},
        {"override /scheduler": "constant"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_teleop_raw",
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
    dataloader_train=dataloader_train_teleop_raw,
    dataloader_val=dataloader_val_teleop_raw,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=7_000,
        run_validation=True,
        validation_iter=200,
        max_val_iter=10,
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
    predict2_video2world_training_2b_teleop_raw,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
