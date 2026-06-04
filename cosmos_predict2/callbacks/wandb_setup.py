import os
from pathlib import Path

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

    def _sanitize_for_wandb(self, value, depth: int = 0):
        """Convert config objects to W&B-safe JSON-ish values."""
        if depth > 8:
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._sanitize_for_wandb(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_wandb(v, depth + 1) for v in value]
        if hasattr(value, "__dict__"):
            return {
                str(k): self._sanitize_for_wandb(v, depth + 1)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        return str(value)

    def _build_wandb_config(self) -> dict:
        # self.config can be a nested lazy/Hydra object; sanitize recursively.
        return self._sanitize_for_wandb(self.config)

    def on_train_start(self, model, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        if not os.environ.get("WANDB_API_KEY"):
            return
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", self.config.job.project),
            entity=os.environ.get("WANDB_ENTITY"),
            group=os.environ.get("WANDB_RUN_GROUP", self.config.job.group),
            name=os.environ.get("WANDB_RUN_ID", self.config.job.name),
            resume="allow",
            config=self._build_wandb_config(),
        )

    def on_app_end(self) -> None:
        if wandb.run:
            wandb.finish()
