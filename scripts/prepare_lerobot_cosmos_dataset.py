# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert a LeRobot v3.0 dataset to VideoDataset format (MP4 + .txt captions).

Camera: the first video key containing "top" (topdown view). Override with --camera.

FPS: source datasets are typically 30 fps. We downsample to 10 fps by keeping every
3rd frame via ffmpeg's fps filter (Option A — frame dropping, no interpolation).
The MIN_FRAMES check is applied to the *output* frame count after downsampling.

Resolution: defaults to 480×640 (H×W). Both dimensions must be divisible by 16
(8× tokenizer spatial compression × 2× DiT patch size). By default the script
checks that the dataset's native resolution matches --video-size and raises an
error if not. Pass --resize to scale to --video-size instead of erroring.

Usage:
    uv run --with pyarrow --with pandas --with tqdm \
        python scripts/prepare_lerobot_cosmos_dataset.py \
        --src datasets/yams_lerobot \
        --out datasets/yams_cosmos_mp4 \
        [--camera top] \
        [--fps 10] \
        [--video-size 480 640] \
        [--resize] \
        [--val-fraction 0.1]

Output structure (identical to libero):
    datasets/yams_cosmos_mp4/
        train/
            videos/   episode_NNNNNN.mp4   (H.264 lossless, 10 fps)
            metas/    episode_NNNNNN.txt   (task description)
        val/
            videos/   episode_NNNNNN.mp4
            metas/    episode_NNNNNN.txt

Next step - generate T5 embeddings:
    python -m scripts.get_t5_embeddings --dataset_path datasets/yams_cosmos_mp4/train
    python -m scripts.get_t5_embeddings --dataset_path datasets/yams_cosmos_mp4/val

num_frames for training: 93
    Aligned with agibot config (state_t=24, temporal_compression_factor=4):
    formula: (state_t - 1) * temporal_compression + 1 = (24-1)*4+1 = 93
    Episodes with fewer than 94 output frames (after fps downsampling) are skipped.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

MIN_OUTPUT_FRAMES = 94  # need output_frames >= 94 so randint(0, max_start_idx) works for num_frames=93


def load_info(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def load_tasks(dataset_dir: Path) -> dict[int, str]:
    t = pq.read_table(str(dataset_dir / "meta" / "tasks.parquet"))
    df = t.to_pandas().reset_index()  # task string is the pandas index
    return dict(zip(df["task_index"], df["task"]))


def build_episode_list(dataset_dir: Path) -> list[dict]:
    """Scan all data parquet files and build one record per episode."""
    episodes = []
    for pq_path in sorted(dataset_dir.glob("data/chunk-*/file-*.parquet")):
        chunk_idx = int(pq_path.parent.name.split("-")[1])
        file_idx = int(pq_path.stem.split("-")[1])

        df = pq.read_table(str(pq_path)).to_pandas()
        file_start_index = int(df["index"].min())

        for ep_idx, group in df.groupby("episode_index"):
            local_start = int(group["index"].min()) - file_start_index
            episodes.append({
                "episode_index": int(ep_idx),
                "chunk_idx": chunk_idx,
                "file_idx": file_idx,
                "local_start": local_start,       # frame offset within the source video file
                "length": len(group),              # source frame count
                "task_index": int(group["task_index"].iloc[0]),
            })

    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def select_camera_key(info: dict, preferred: str) -> str:
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    matches = [k for k in video_keys if preferred.lower() in k.lower()]
    if matches:
        return matches[0]
    if video_keys:
        print(f"WARNING: no key matching '{preferred}', falling back to '{video_keys[0]}'")
        return video_keys[0]
    raise ValueError(f"No video keys in dataset features. Available: {list(info['features'])}")


def get_source_resolution(info: dict, camera_key: str) -> tuple[int, int]:
    """Return (H, W) from the dataset's feature shape for the given camera key."""
    shape = info["features"][camera_key]["shape"]  # [H, W, C]
    return shape[0], shape[1]


def check_resolution(src_h: int, src_w: int, target_h: int, target_w: int, resize: bool) -> None:
    if src_h == target_h and src_w == target_w:
        return
    if resize:
        print(f"Source resolution {src_h}×{src_w} differs from target {target_h}×{target_w} — resizing.")
        return
    raise ValueError(
        f"Source resolution {src_h}×{src_w} does not match --video-size {target_h}×{target_w}. "
        f"Pass --resize to scale instead of erroring."
    )


def get_source_video(dataset_dir: Path, video_path_template: str, camera_key: str, chunk_idx: int, file_idx: int) -> Path:
    rel = video_path_template.format(video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx)
    return dataset_dir / rel


def convert_episode(
    src_video: Path,
    out_dir: Path,
    episode_name: str,
    task_description: str,
    local_start: int,
    source_length: int,
    video_size: tuple[int, int],
    target_fps: int,
    resize: bool,
) -> bool:
    H, W = video_size
    out_video = out_dir / "videos" / f"{episode_name}.mp4"

    # Extract the episode segment and downsample fps in one pass.
    # trim:        frame-accurate extraction of the episode from the source file
    # setpts:      reset timestamps to 0 after trim
    # fps=N:       drop frames to reach target fps (every Kth frame kept, no interpolation)
    # scale:       only added when --resize is set
    vf = (
        f"trim=start_frame={local_start}:end_frame={local_start + source_length},"
        f"setpts=PTS-STARTPTS,"
        f"fps={target_fps}"
    )
    if resize:
        vf += f",scale={W}:{H}"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src_video),
            "-vf", vf,
            "-c:v", "libx264",
            "-crf", "0",            # lossless H.264
            "-preset", "ultrafast",
            "-an",                  # strip audio
            str(out_video),
        ],
        check=True,
        capture_output=True,
    )

    # ffmpeg exits 0 even when trim produces 0 frames — validate the output has a real video stream.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "default=noprint_wrappers=1:nokey=1",
         str(out_video)],
        capture_output=True, text=True,
    )
    nb_frames = int(probe.stdout.strip() or 0)
    if nb_frames < MIN_OUTPUT_FRAMES:
        out_video.unlink(missing_ok=True)
        raise ValueError(f"Output has only {nb_frames} frames (expected >= {MIN_OUTPUT_FRAMES})")

    (out_dir / "metas" / f"{episode_name}.txt").write_text(task_description)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="LeRobot v3.0 dataset root directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--camera", type=str, default="top", help="Substring to match camera key (default: top)")
    parser.add_argument("--fps", type=int, default=10, help="Output fps (default: 10)")
    parser.add_argument("--video-size", type=int, nargs=2, default=[480, 640], metavar=("H", "W"),
                        help="Expected/output size H W (default: 480 640). Both must be divisible by 16.")
    parser.add_argument("--resize", action="store_true",
                        help="Resize to --video-size instead of raising an error on resolution mismatch.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    H, W = args.video_size
    assert H % 16 == 0 and W % 16 == 0, f"video_size ({H}, {W}) must both be divisible by 16"

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    info = load_info(src_dir)
    tasks = load_tasks(src_dir)
    src_fps = info["fps"]
    video_path_template = info["video_path"]

    camera_key = select_camera_key(info, args.camera)
    src_h, src_w = get_source_resolution(info, camera_key)
    check_resolution(src_h, src_w, H, W, args.resize)

    print(f"Camera key:      {camera_key}")
    print(f"Source FPS:      {src_fps}  →  output FPS: {args.fps}")
    print(f"Source size:     {src_h}×{src_w}  →  output size: {H}×{W}")

    episodes = build_episode_list(src_dir)
    print(f"Total episodes:  {len(episodes)}")

    for split in ("train", "val"):
        (out_dir / split / "videos").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "metas").mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    shuffled = [ep["episode_index"] for ep in episodes]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_set = set(shuffled[:n_val])

    converted_train, converted_val, skipped = 0, 0, 0
    for ep in tqdm(episodes):
        # Check length after fps downsampling
        output_frames = ep["length"] * args.fps // src_fps
        if output_frames < MIN_OUTPUT_FRAMES:
            tqdm.write(f"  SKIP episode {ep['episode_index']}: {output_frames} output frames < {MIN_OUTPUT_FRAMES}")
            skipped += 1
            continue

        src_video = get_source_video(src_dir, video_path_template, camera_key, ep["chunk_idx"], ep["file_idx"])
        if not src_video.exists():
            tqdm.write(f"  SKIP episode {ep['episode_index']}: video not found: {src_video}")
            skipped += 1
            continue

        split = "val" if ep["episode_index"] in val_set else "train"
        task_desc = tasks.get(ep["task_index"], "robot manipulation task")
        episode_name = f"episode_{ep['episode_index']:06d}"

        try:
            convert_episode(
                src_video, out_dir / split, episode_name, task_desc,
                ep["local_start"], ep["length"], (H, W), args.fps, args.resize,
            )
        except subprocess.CalledProcessError as e:
            tqdm.write(f"  ERROR episode {ep['episode_index']}: {e.stderr.decode()[-300:]}")
            skipped += 1
            continue

        if split == "val":
            converted_val += 1
        else:
            converted_train += 1

    print(f"\nDone. Train: {converted_train}, Val: {converted_val}, Skipped: {skipped}")
    print(f"Output: {out_dir}")
    print(f"\nVideoDataset config:")
    print(f"  dataset_dir = '{out_dir}/train'  (train)")
    print(f"  dataset_dir = '{out_dir}/val'    (val)")
    print(f"  video_size  = ({H}, {W})")
    print(f"\nNext: generate T5 embeddings:")
    print(f"  python -m scripts.get_t5_embeddings --dataset_path {out_dir}/train")
    print(f"  python -m scripts.get_t5_embeddings --dataset_path {out_dir}/val")


if __name__ == "__main__":
    main()
