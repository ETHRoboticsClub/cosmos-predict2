import os
from typing import Any

import wandb

from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback


class WandbSetup(Callback):
    """Initialize and tear down a W&B run around training.

    Activated only when WANDB_API_KEY is set in the environment.
    Project / group / name are pulled from the job config but can be
    overridden with the standard WANDB_PROJECT / WANDB_RUN_GROUP /
    WANDB_RUN_ID env vars.
    """

    @staticmethod
    def _lookup(config: Any, path: str, default: Any = None) -> Any:
        value = config
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part, default)
            else:
                value = getattr(value, part, default)
            if value is default:
                return default
        return value

    @staticmethod
    def _as_wandb_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
            return list(value)
        return str(value)

    def _wandb_config(self, iteration: int) -> dict[str, Any]:
        keys = [
            "job.project",
            "job.group",
            "job.name",
            "trainer.distributed_parallelism",
            "trainer.max_iter",
            "trainer.validation_iter",
            "trainer.max_val_iter",
            "checkpoint.save_iter",
            "optimizer.lr",
            "optimizer.weight_decay",
            "model_parallel.context_parallel_size",
            "dataloader_train.batch_size",
            "dataloader_train.num_workers",
            "dataloader_train.dataset.dataset_dir",
            "dataloader_train.dataset.camera_name",
            "dataloader_train.dataset.num_frames",
            "dataloader_train.dataset.video_size",
            "dataloader_train.dataset.embedding_cache_dir",
            "dataloader_val.batch_size",
            "dataloader_val.num_workers",
            "model.config.train_architecture",
            "model.config.lora_rank",
            "model.config.lora_alpha",
            "model.config.lora_target_modules",
            "model.config.fsdp_shard_size",
            "model.config.precision",
            "model.config.model_manager_config.dit_path",
        ]
        config = {"trainer.start_iteration": iteration}
        for key in keys:
            value = self._lookup(self.config, key)
            if value is not None:
                config[key] = self._as_wandb_value(value)
        return config

    def on_train_start(self, model, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        if not os.environ.get("WANDB_API_KEY"):
            return
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", self.config.job.project),
            group=os.environ.get("WANDB_RUN_GROUP", self.config.job.group),
            name=os.environ.get("WANDB_RUN_ID", self.config.job.name),
            resume="allow",
            config=self._wandb_config(iteration),
        )

    def on_app_end(self) -> None:
        if wandb.run:
            wandb.finish()
