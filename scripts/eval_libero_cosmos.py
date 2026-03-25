# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate Cosmos Predict2 on LIBERO val episodes.

Picks N random val videos, runs inference with 5 context frames (mimic-video paper),
and saves a side-by-side comparison: ground truth (left) | generated (right).

Run twice to compare base vs. fine-tuned:

    # Base model
    python scripts/eval_libero_cosmos.py --out eval/base

    # Fine-tuned LoRA
    python scripts/eval_libero_cosmos.py --out eval/finetuned \\
        --lora-checkpoint /path/to/checkpoints/<iter>/model_ema_bf16.pt
"""

import argparse
import os
import random
import subprocess
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from tqdm import tqdm

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video

_MODEL_SIZE = "2B"
_RESOLUTION = "480"
_FPS = 10
_ASPECT_RATIO = "1:1"
_NUM_CONDITIONAL_FRAMES = 5  # mimic-video paper
_LORA_RANK = 16
_LORA_ALPHA = 16
_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"


def load_pipeline(lora_checkpoint: str | None) -> Video2WorldPipeline:
    config = get_cosmos_predict2_video2world_pipeline(model_size=_MODEL_SIZE, resolution=_RESOLUTION, fps=_FPS)
    config.prompt_refiner_config.enabled = False
    config.guardrail_config.enabled = False

    dit_path = get_cosmos_predict2_video2world_checkpoint(model_size=_MODEL_SIZE, resolution=_RESOLUTION, fps=_FPS)
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    if lora_checkpoint:
        print(f"Injecting LoRA and loading fine-tuned weights from {lora_checkpoint}")
        from peft import LoraConfig, inject_adapter_in_model

        from cosmos_predict2.models.utils import load_state_dict

        lora_cfg = LoraConfig(
            r=_LORA_RANK,
            lora_alpha=_LORA_ALPHA,
            init_lora_weights=True,
            target_modules=_LORA_TARGET_MODULES.split(","),
        )
        pipe.dit = inject_adapter_in_model(lora_cfg, pipe.dit)

        state_dict = load_state_dict(lora_checkpoint)
        # Prefer EMA weights; fall back to regular weights
        weights = {k[8:]: v for k, v in state_dict.items() if k.startswith("net_ema.")}
        if not weights:
            weights = {k[4:]: v for k, v in state_dict.items() if k.startswith("net.")}
        pipe.dit.load_state_dict(weights, strict=False)
        pipe.dit = pipe.dit.to(device="cuda", dtype=torch.bfloat16)

    return pipe


def save_gt_clip(src: Path, dst: Path, size: int = 480) -> None:
    """Resize ground truth to match generated resolution for side-by-side."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"scale={size}:{size}",
            "-c:v", "libx264", "-crf", "18",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def make_comparison(gt_path: Path, gen_path: Path, out_path: Path) -> None:
    """Horizontally stack ground truth and generated video."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(gt_path),
            "-i", str(gen_path),
            "-filter_complex", "hstack=inputs=2",
            "-c:v", "libx264", "-crf", "18",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default="datasets/libero_cosmos_mp4/val")
    parser.add_argument("--out", default="eval/base", help="Output directory")
    parser.add_argument("--lora-checkpoint", default=None, help="Path to model_ema_bf16.pt (omit for base model)")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed — same episodes are sampled across runs")
    parser.add_argument("--guidance", type=float, default=7.0)
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_videos = sorted((val_dir / "videos").glob("*.mp4"))
    if not all_videos:
        raise FileNotFoundError(f"No videos found in {val_dir / 'videos'}")

    rng = random.Random(args.seed)
    videos = rng.sample(all_videos, min(args.num_videos, len(all_videos)))
    print(f"Sampled {len(videos)} episodes from {val_dir}")

    pipe = load_pipeline(args.lora_checkpoint)

    for video_path in tqdm(videos):
        name = video_path.stem
        caption_path = val_dir / "metas" / f"{name}.txt"
        prompt = caption_path.read_text().strip() if caption_path.exists() else "robot manipulation task"

        print(f"\n{name}: {prompt}")

        video = pipe(
            input_path=str(video_path),
            prompt=prompt,
            aspect_ratio=_ASPECT_RATIO,
            num_conditional_frames=_NUM_CONDITIONAL_FRAMES,
            guidance=args.guidance,
            seed=args.seed,
        )
        if video is None:
            print(f"  Skipped (pipeline returned None)")
            continue

        gen_path = out_dir / f"{name}_generated.mp4"
        gt_path = out_dir / f"{name}_gt.mp4"
        comparison_path = out_dir / f"{name}_comparison.mp4"

        save_image_or_video(video, str(gen_path), fps=_FPS)
        save_gt_clip(video_path, gt_path)
        make_comparison(gt_path, gen_path, comparison_path)
        print(f"  Saved: {comparison_path}")

    print(f"\nDone. All results in {out_dir}/")


if __name__ == "__main__":
    main()