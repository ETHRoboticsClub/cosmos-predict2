# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Download a single episode from a LeRobot HuggingFace dataset and run
Video2World inference at regular time intervals throughout the episode.

Examples:

    # Base model, every 5 seconds (default)
    python scripts/run_lerobot_inference.py \
        --repo_id ETHRC/yams-closed-carton-box-to-migros-basket-go2 \
        --episode 0 \
        --dit_path checkpoints/Cosmos-Predict2-2B-Video2World/model.pt

    # LoRA fine-tuned, 5-frame conditioning, every 3 seconds
    python scripts/run_lerobot_inference.py \
        --repo_id ETHRC/yams-closed-carton-box-to-migros-basket-go2 \
        --episode 0 \
        --interval_sec 3 \
        --num_conditional_frames 5 \
        --dit_path checkpoints/Cosmos-Predict2-2B-Video2World/model.pt \
        --lora_checkpoint outputs/checkpoints/model/iter_007000.pt
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pyarrow.parquet as pq
import torch

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils.io import save_image_or_video


# ---------------------------------------------------------------------------
# LeRobot metadata helpers (adapted from prepare_lerobot_cosmos_dataset.py)
# ---------------------------------------------------------------------------

def load_info(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def load_tasks(dataset_dir: Path) -> dict[int, str]:
    t = pq.read_table(str(dataset_dir / "meta" / "tasks.parquet"))
    df = t.to_pandas().reset_index()
    return dict(zip(df["task_index"], df["task"]))


def select_camera_key(info: dict, preferred: str) -> str:
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    matches = [k for k in video_keys if preferred.lower() in k.lower()]
    if matches:
        return matches[0]
    if video_keys:
        print(f"WARNING: no key matching '{preferred}', falling back to '{video_keys[0]}'")
        return video_keys[0]
    raise ValueError(f"No video keys in dataset features. Available: {list(info['features'])}")


def get_episode_video_path(info: dict, camera_key: str, episode: int) -> str:
    """Return the relative path to the video file for a given episode."""
    template = info["video_path"]
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode // chunks_size
    return template.format(video_key=camera_key, chunk_index=chunk_idx, file_index=episode)


def get_task_for_episode(dataset_dir: Path, episode: int) -> str:
    """Read task description for a specific episode from the data parquet files."""
    tasks = load_tasks(dataset_dir)

    # Scan data parquets to find the task_index for this episode
    for pq_path in sorted(dataset_dir.glob("data/chunk-*/file-*.parquet")):
        df = pq.read_table(str(pq_path)).to_pandas()
        match = df[df["episode_index"] == episode]
        if not match.empty:
            task_index = int(match["task_index"].iloc[0])
            return tasks.get(task_index, "robot manipulation task")

    print(f"WARNING: episode {episode} not found in data parquets, using fallback prompt")
    return "robot manipulation task"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_episode(repo_id: str, dataset_dir: Path, camera_key: str, video_rel: str, episode: int) -> None:
    """Download metadata + one episode video from HuggingFace. Skips if already present."""
    from huggingface_hub import snapshot_download

    video_path = dataset_dir / video_rel
    if video_path.exists():
        print(f"Episode video already exists: {video_path}")
        return

    # Step 1: metadata
    print(f"Downloading metadata from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_dir),
        allow_patterns=["meta/info.json", "meta/tasks.parquet"],
    )

    # Step 2: the data parquet containing this episode (for task description)
    chunks_size = load_info(dataset_dir).get("chunks_size", 1000)
    chunk_idx = episode // chunks_size
    data_pattern = f"data/chunk-{chunk_idx:03d}/*.parquet"
    print(f"Downloading data parquet ({data_pattern})...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_dir),
        allow_patterns=[data_pattern],
    )

    # Step 3: the episode video
    print(f"Downloading episode video ({video_rel})...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_dir),
        allow_patterns=[video_rel],
    )
    print(f"Downloaded to {video_path}")


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def get_video_frame_count(video_path: Path) -> int:
    """Return total frame count of a video file."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def inference_times(duration_sec: float, interval_sec: float) -> list[float]:
    """Return timestamps (seconds) at which to run inference."""
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be positive, got {interval_sec}")
    times: list[float] = []
    t = 0.0
    while t < duration_sec:
        times.append(t)
        t += interval_sec
    return times


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    """Extract a single frame as PNG."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_index})",
            "-vframes", "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    print(f"Extracted frame {frame_index} to {output_path}")


def extract_clip(video_path: Path, start_frame: int, num_frames: int, output_path: Path) -> None:
    """Extract a short clip for multi-frame conditioning."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", (
                f"trim=start_frame={start_frame}"
                f":end_frame={start_frame + num_frames},"
                f"setpts=PTS-STARTPTS"
            ),
            "-c:v", "libx264", "-crf", "18",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    print(f"Extracted {num_frames}-frame clip (from frame {start_frame}) to {output_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_pipeline(
    dit_path: str,
    lora_checkpoint: str | None,
    model_size: str,
    resolution: str,
    fps: int,
) -> Video2WorldPipeline:
    config = get_cosmos_predict2_video2world_pipeline(model_size=model_size, resolution=resolution, fps=fps)
    config.prompt_refiner_config.enabled = False
    config.guardrail_config.enabled = False

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
            r=16,
            lora_alpha=16,
            init_lora_weights=True,
            target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"],
        )
        pipe.dit = inject_adapter_in_model(lora_cfg, pipe.dit)

        state_dict = load_state_dict(lora_checkpoint)
        weights = {k[8:]: v for k, v in state_dict.items() if k.startswith("net_ema.")}
        if not weights:
            weights = {k[4:]: v for k, v in state_dict.items() if k.startswith("net.")}
        pipe.dit.load_state_dict(weights, strict=False)
        pipe.dit = pipe.dit.to(device="cuda", dtype=torch.bfloat16)

    return pipe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a LeRobot episode and run Video2World inference")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--interval_sec",
        type=float,
        default=5.0,
        help="Run inference every N seconds across the episode (default: 5)",
    )
    parser.add_argument("--camera", type=str, default="top", help="Substring to match camera key")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to base DiT checkpoint")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to LoRA checkpoint .pt")
    parser.add_argument("--prompt", type=str, default=None, help="Override text prompt")
    parser.add_argument("--model_size", type=str, default="2B", help="Model size")
    parser.add_argument("--resolution", type=str, default="480", help="Pipeline resolution")
    parser.add_argument("--fps", type=int, default=10, help="Pipeline FPS")
    parser.add_argument("--aspect_ratio", type=str, default="1:1", help="Output aspect ratio")
    parser.add_argument("--num_conditional_frames", type=int, default=1, choices=[1, 5], help="Conditioning frames")
    parser.add_argument("--guidance", type=float, default=7.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- 1. Download episode ---
    dataset_dir = Path("datasets") / args.repo_id.replace("/", "_")
    info_path = dataset_dir / "meta" / "info.json"

    # We need info.json first to resolve the camera key and video path.
    # If it doesn't exist yet, download metadata.
    if not info_path.exists():
        from huggingface_hub import snapshot_download

        print(f"Downloading metadata from {args.repo_id}...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=["meta/info.json", "meta/tasks.parquet"],
        )

    info = load_info(dataset_dir)
    camera_key = select_camera_key(info, args.camera)
    video_rel = get_episode_video_path(info, camera_key, args.episode)

    print(f"Camera key:    {camera_key}")
    print(f"Video path:    {video_rel}")

    download_episode(args.repo_id, dataset_dir, camera_key, video_rel, args.episode)

    episode_video = dataset_dir / video_rel

    # --- 2. Get prompt ---
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = get_task_for_episode(dataset_dir, args.episode)
    print(f"Prompt:        {prompt}")

    # --- 3. Schedule inference timestamps ---
    src_fps = info.get("fps", args.fps)
    duration_sec = get_video_duration(episode_video)
    total_frames = get_video_frame_count(episode_video)
    times = inference_times(duration_sec, args.interval_sec)

    print(f"Episode FPS:   {src_fps}")
    print(f"Duration:      {duration_sec:.2f}s ({total_frames} frames)")
    print(f"Interval:      {args.interval_sec}s → {len(times)} inference(s)")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 4. Load pipeline once ---
    pipe = load_pipeline(
        dit_path=args.dit_path,
        lora_checkpoint=args.lora_checkpoint,
        model_size=args.model_size,
        resolution=args.resolution,
        fps=args.fps,
    )

    # --- 5. Run inference at each timestamp ---
    saved = 0
    for time_sec in times:
        frame_index = int(time_sec * src_fps)
        if frame_index + args.num_conditional_frames > total_frames:
            print(
                f"Skipping t={time_sec:.1f}s (frame {frame_index}): "
                f"need {args.num_conditional_frames} frames, only {total_frames - frame_index} left"
            )
            continue

        tag = f"{int(time_sec):04d}s"
        print(f"\n--- t={time_sec:.1f}s (frame {frame_index}) ---")

        if args.num_conditional_frames == 1:
            input_path = os.path.join(args.output_dir, f"extracted_frame_{tag}.png")
            extract_frame(episode_video, frame_index, Path(input_path))
        else:
            input_path = os.path.join(args.output_dir, f"extracted_clip_{tag}.mp4")
            extract_clip(episode_video, frame_index, args.num_conditional_frames, Path(input_path))

        print("Running inference...")
        video = pipe(
            input_path=input_path,
            prompt=prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
        )

        if video is not None:
            output_path = os.path.join(args.output_dir, f"generated_{tag}.mp4")
            save_image_or_video(video, output_path, fps=args.fps)
            print(f"Saved generated video to {output_path}")
            saved += 1
        else:
            print(f"Pipeline returned None at t={time_sec:.1f}s — generation failed.")

    print(f"\nDone. Saved {saved}/{len(times)} generated video(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
