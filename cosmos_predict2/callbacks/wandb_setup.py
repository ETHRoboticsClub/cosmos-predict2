import os

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
        )

    def on_app_end(self) -> None:
        if wandb.run:
            wandb.finish()