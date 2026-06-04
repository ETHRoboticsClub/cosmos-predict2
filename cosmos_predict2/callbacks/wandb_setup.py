import os
import tempfile
from pathlib import Path

import wandb
import yaml

from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed
from imaginaire.utils import log
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
        # Prefer the same resolved YAML that the trainer writes locally. It
        # includes defaults/overrides and is already converted to plain values.
        resolved_config = self._load_resolved_config()
        if resolved_config is None:
            # self.config can be a nested lazy/Hydra object; sanitize recursively.
            resolved_config = self._sanitize_for_wandb(self.config)
        return self._flatten_config(resolved_config)

    def _load_resolved_config(self) -> dict | None:
        config_path = Path(self.config.job.path_local) / "config.yaml"
        try:
            if not config_path.exists():
                with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as tmp:
                    LazyConfig.save_yaml(self.config, tmp.name)
                    tmp.seek(0)
                    return yaml.unsafe_load(tmp) or {}
            with config_path.open() as f:
                return yaml.unsafe_load(f) or {}
        except Exception as exc:
            log.warning(f"Failed to load resolved config for W&B; falling back to object sanitizer: {exc}")
            return None

    def _flatten_config(self, value, prefix: str = "") -> dict:
        """Flatten nested config values so W&B UI/search exposes every leaf."""
        if isinstance(value, dict):
            flattened = {}
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                flattened.update(self._flatten_config(child, child_prefix))
            return flattened
        if isinstance(value, list):
            if not value:
                return {prefix: []}
            flattened = {}
            for index, child in enumerate(value):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                flattened.update(self._flatten_config(child, child_prefix))
            return flattened
        return {prefix: self._sanitize_for_wandb(value)}

    def on_train_start(self, model, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        if not os.environ.get("WANDB_API_KEY"):
            return
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", self.config.job.project),
            entity=os.environ.get("WANDB_ENTITY"),
            group=os.environ.get("WANDB_RUN_GROUP", self.config.job.group),
            name=os.environ.get("WANDB_RUN_ID", self.config.job.name),
            resume="allow",
            config=self._build_wandb_config(),
        )
        config_path = Path(self.config.job.path_local) / "config.yaml"
        if config_path.exists():
            run.save(str(config_path), base_path=str(config_path.parent), policy="now")

    def on_app_end(self) -> None:
        if wandb.run:
            wandb.finish()
