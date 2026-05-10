# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate Cosmos Predict2 on LIBERO val episodes.

For each episode, runs two comparisons:
  - from_start:  condition on first 5 frames, GT = first 93 frames
  - from_middle: condition on 5 frames at the midpoint, GT = 93 frames from midpoint

Run twice to compare base vs. fine-tuned:

    # Base model
    python scripts/eval_libero_cosmos.py --out eval/base

    # Fine-tuned LoRA
    python scripts/eval_libero_cosmos.py --out eval/finetuned \\
        --lora-checkpoint /path/to/checkpoints/model/<iter>.pt
"""

import argparse
import copy
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
import pickle
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use system ffmpeg if available, fall back to imageio_ffmpeg bundle.
_FFMPEG = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()

import torch
from tqdm import tqdm

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video

_MODEL_SIZE = "2B"
_FPS = 10
_ASPECT_RATIO = "1:1"
_NUM_CONDITIONAL_FRAMES = 5  # mimic-video paper
_LORA_RANK = 16
_LORA_ALPHA = 16
_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"


def load_pipeline(lora_checkpoint: str | None, resolution: str = "480") -> Video2WorldPipeline:
    # "240" has no dedicated checkpoint — load the 480p model and override resolution for inference sizing.
    model_resolution = resolution if resolution in ("480", "720") else "480"
    config = copy.deepcopy(
        get_cosmos_predict2_video2world_pipeline(model_size=_MODEL_SIZE, resolution=model_resolution, fps=_FPS)
    )
    config.prompt_refiner_config.enabled = False
    config.guardrail_config.enabled = False
    if resolution != model_resolution:
        config.resolution = resolution  # generates at the requested smaller size

    dit_path = get_cosmos_predict2_video2world_checkpoint(model_size=_MODEL_SIZE, resolution=model_resolution, fps=_FPS)
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        use_text_encoder=False,  # T5-11B is 43 GB; use pre-computed embeddings from .pickle files instead
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


def get_video_frame_count(path: Path) -> int:
    """Return total frame count of a video file using decord."""
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return len(vr)


def make_cond_clip(src: Path, dst: Path, start_frame: int, num_frames: int) -> None:
    """Extract [start_frame, start_frame+num_frames) as a standalone clip for pipe() conditioning."""
    subprocess.run(
        [
            _FFMPEG, "-y", "-i", str(src),
            "-vf", f"trim=start_frame={start_frame}:end_frame={start_frame + num_frames},setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-crf", "18",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def save_gt_clip(src: Path, dst: Path, size: int = 480, start_frame: int = 0, num_frames: int | None = None) -> None:
    """Trim to [start_frame, start_frame+num_frames), resize to size×size, and save."""
    vf_parts = []
    if start_frame > 0 or num_frames is not None:
        trim = f"trim=start_frame={start_frame}"
        if num_frames is not None:
            trim += f":end_frame={start_frame + num_frames}"
        trim += ",setpts=PTS-STARTPTS"
        vf_parts.append(trim)
    vf_parts.append(f"scale={size}:{size}")
    subprocess.run(
        [_FFMPEG, "-y", "-i", str(src), "-vf", ",".join(vf_parts), "-c:v", "libx264", "-crf", "18", str(dst)],
        check=True,
        capture_output=True,
    )


def make_comparison(gt_path: Path, gen_path: Path, out_path: Path) -> None:
    """Horizontally stack ground truth and generated video."""
    subprocess.run(
        [
            _FFMPEG, "-y",
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
    parser.add_argument("--lora-checkpoint", default=None, help="Path to model checkpoint .pt (omit for base model)")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed — same episodes are sampled across runs")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--resolution", default="480", choices=["240", "480", "512", "720"],
                        help="Output resolution (240 for fast low-res test, 480 for full quality)")
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

    pipe = load_pipeline(args.lora_checkpoint, resolution=args.resolution)

    for video_path in tqdm(videos):
        name = video_path.stem
        caption_path = val_dir / "metas" / f"{name}.txt"
        prompt = caption_path.read_text().strip() if caption_path.exists() else "robot manipulation task"
        print(f"\n{name}: {prompt}")

        # Load pre-computed T5 embedding from dataset (avoids running the text encoder).
        # Pickles store trimmed embeddings [n_tokens, 1024]; pad to 512 with zeros to match
        # what CosmosT5TextEncoder.encode_prompts() returns — the DIT was always trained on
        # 512-length sequences and the cross-attention softmax depends on that fixed length.
        _T5_NUM_TOKENS = 512
        _T5_EMBED_DIM = 1024
        t5_pickle = val_dir / "t5_xxl" / f"{name}.pickle"
        if t5_pickle.exists():
            with open(t5_pickle, "rb") as f:
                t5_raw = pickle.load(f)
            raw = torch.from_numpy(np.array(t5_raw[0])).float()  # [n_tokens, 1024]
            n_tokens = raw.shape[0]
            t5_embeddings = torch.zeros(_T5_NUM_TOKENS, _T5_EMBED_DIM)
            t5_embeddings[:n_tokens] = raw  # zero-pad to 512, matching encode_prompts output
        else:
            print(f"  Warning: no T5 embedding at {t5_pickle}, using zero embedding for debugging")
            t5_embeddings = torch.zeros(_T5_NUM_TOKENS, _T5_EMBED_DIM)

        total_frames = get_video_frame_count(video_path)

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            for mode in ("from_start", "from_middle"):
                if mode == "from_start":
                    cond_start = 0
                else:
                    # Middle: centre the conditioning window, ensure enough GT frames remain
                    cond_start = max(0, total_frames // 2 - _NUM_CONDITIONAL_FRAMES // 2)

                # Extract exactly _NUM_CONDITIONAL_FRAMES frames so pipe() conditions on the right segment.
                # (read_and_process_video always takes the last N frames of the input clip)
                cond_clip = tmp / f"cond_{mode}.mp4"
                make_cond_clip(video_path, cond_clip, start_frame=cond_start, num_frames=_NUM_CONDITIONAL_FRAMES)

                video = pipe(
                    input_path=str(cond_clip),
                    prompt=prompt,
                    aspect_ratio=_ASPECT_RATIO,
                    num_conditional_frames=_NUM_CONDITIONAL_FRAMES,
                    guidance=args.guidance,
                    seed=args.seed,
                    t5_embeddings=t5_embeddings.cuda().to(torch.bfloat16),
                )
                if video is None:
                    print(f"  Skipped {mode} (pipeline returned None)")
                    continue

                num_gen_frames = video.shape[2]

                # GT starts at the same frame as conditioning — same window for both modes.
                gt_start = cond_start

                gt_path = out_dir / f"{name}_{mode}_gt.mp4"
                gen_path = out_dir / f"{name}_{mode}_gen.mp4"
                comparison_path = out_dir / f"{name}_{mode}_comparison.mp4"

                save_gt_clip(video_path, gt_path, size=int(args.resolution), start_frame=gt_start, num_frames=num_gen_frames)
                save_image_or_video(video, str(gen_path), fps=_FPS)
                make_comparison(gt_path, gen_path, comparison_path)
                print(f"  Saved ({mode}): {comparison_path}")

    print(f"\nDone. All results in {out_dir}/")


if __name__ == "__main__":
    main()
