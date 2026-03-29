# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state

from cosmos_predict2.utils.context_parallel import cat_outputs_cp
from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from imaginaire.visualize.video import save_img_or_video

# from imaginaire.visualize.video import save_img_or_video
# from projects.cosmos.diffusion.v2.datasets.data_sources.item_datasets_for_validation import get_itemdataset_option


# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: str | None = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: list[float] = [3.0, 7.0, 9.0, 13.0],  # noqa: B006
        is_x0: bool = True,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        show_all_frames: bool = False,
        fps: int = 16,
    ):
        super().__init__(every_n, step_size)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_x0 = is_x0
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.show_all_frames = show_all_frames
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self._first_val_batch = None

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if self.fix_batch is not None:
            with misc.timer(f"loading fix_batch {self.fix_batch}"):
                self.fix_batch = misc.co(easy_io.load(self.fix_batch), "cpu")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        # if self.use_negative_prompt:
        #     self.negative_prompt_data = easy_io.load(get_itemdataset_option("negative_prompt_v0_s3").path)

    def on_validation_start(self, model: ImaginaireModel, dataloader_val, iteration: int = 0) -> None:
        self._first_val_batch = None

    @torch.no_grad()
    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict,
        output_batch: dict,
        loss,
        iteration: int = 0,
    ) -> None:
        # Capture only the first val batch; we're already inside ema_scope from the trainer.
        if self._first_val_batch is not None:
            return
        self._first_val_batch = data_batch
        if not self.is_x0:
            return
        # ema_scope is already active — use nullcontext to avoid re-entering it.
        x0_img_fp, mse_loss, sigmas = self.x0_pred(None, model, data_batch, output_batch, loss, iteration)
        dist.barrier()
        if wandb.run:
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag = f"val_ema_{data_type}"
            info = {
                "trainer/global_step": iteration,
                f"{self.name}/{tag}_x0": self._to_wandb_media(x0_img_fp, caption=str(iteration)),
            }
            mse_loss = mse_loss.tolist()
            info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})
            wandb.log(info, step=iteration)
        torch.cuda.empty_cache()


    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)
        # TODO: (qsh 2024-07-01) this may be problematic due to sometimes we have uncondition, some times we have condition due to cfg dropout
        # TODO: (qsh 2025-02-25) we need to broadcast raw_data for correct visualization
        raw_data, x0, condition = model.pipe.get_data_and_condition(data_batch)
        _, condition, x0, _ = model.pipe.broadcast_split_for_model_parallelsim(None, condition, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(
                math.log(model.pipe.scheduler.config.sigma_min),
                math.log(model.pipe.scheduler.config.sigma_max),
                self.n_x0_level + 1,
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.pipe.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)
            # Gather CP-split latent from all ranks before decoding to reconstruct the full video.
            if model.pipe.dit.is_context_parallel_enabled:
                sample = cat_outputs_cp(sample, seq_dim=2, cp_group=model.pipe.get_context_parallel_group())
            if hasattr(model.pipe, "decode"):
                sample = model.pipe.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.pipe_config.ema.enabled:
                return
            context = partial(model.pipe.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            log.debug("entering, ema", rank0_only=False)
            # we only use rank0 and rank to generate images and save
            # other rank run forward pass to make sure it works for FSDP
            log.debug("entering, fsdp", rank0_only=False)
            if self.is_x0:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, x0_pred", rank0_only=False)
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )
            if self.is_sample:
                log.debug("entering, sample", rank0_only=False)
                sample_img_fp = self.sample(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, sample", rank0_only=False)
            if self.fix_batch is not None:
                misc.to(self.fix_batch, "cpu")

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.is_x0:
                info[f"{self.name}/{tag}_x0"] = self._to_wandb_media(x0_img_fp, caption=str(sample_counter))
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            if self.is_sample:
                info[f"{self.name}/{tag}_sample"] = self._to_wandb_media(sample_img_fp, caption=str(sample_counter))
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.pipe.get_data_and_condition(data_batch)
        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            data_batch["neg_t5_text_embeddings"] = misc.to(
                repeat(
                    self.negative_prompt_data["t5_text_embeddings"],
                    "... -> b ...",
                    b=batch_size,
                ),
                **model.tensor_kwargs,
            )
            assert data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape, (
                f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            )
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

        # Use only the first guidance value to avoid 4x cost (default list has 4 values).
        # self.guidance may be a Hydra ListConfig (not a plain list), so index then cast.
        guidance = float(self.guidance[0]) if hasattr(self.guidance, "__getitem__") else float(self.guidance)
        # generate_samples_from_batch handles CP split/gather and decode internally.
        sample = model.pipe.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            state_shape=x0.shape[1:],
            n_sample=1,  # only one sample needed for visualization
            num_steps=self.num_sampling_step,
        )
        to_show = [sample.float().cpu(), raw_data[:1].float().cpu()]

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size=1, base_fp_wo_ext=base_fp_wo_ext)
            return local_path
        return None

    def _to_wandb_media(self, media: str | None, caption: str = "") -> wandb.Image | wandb.Video | None:
        """Return wandb.Video for .mp4 paths, wandb.Image for all other paths."""
        if media is None:
            return None
        if media.endswith(".mp4"):
            return wandb.Video(media, fps=self.fps, format="mp4")
        return wandb.Image(media, caption=caption)

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> str | np.ndarray | None:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        if not (self.rank == 0 and wandb.run):
            return None

        to_show = to_show[:, :n_viz_sample]

        if is_single_frame or not self.show_all_frames:
            # 3-frame grid (or single image) → JPEG
            if not is_single_frame:
                _T = to_show.shape[3]
                to_show = to_show[:, :, :, [0, _T // 2, _T - 1]]
                to_show = rearrange(to_show, "n b c t h w -> 1 c (n h) (b t w)")
            else:
                to_show = rearrange(to_show, "n b c t h w -> t c (n h) (b w)")
            image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
            local_path = f"{self.local_dir}/{base_fp_wo_ext}_resize.jpg"
            torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            return local_path
        else:
            # All frames → mp4 file (avoids moviepy dependency for wandb.Video)
            video = rearrange(to_show, "n b c t h w -> t (n h) (b w) c")
            video_uint8 = (video * 255).byte()
            local_path = f"{self.local_dir}/{base_fp_wo_ext}.mp4"
            torchvision.io.write_video(local_path, video_uint8, fps=self.fps)
            return local_path
