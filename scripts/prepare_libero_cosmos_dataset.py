# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert nvidia/LIBERO-Cosmos-Policy HDF5 episodes to VideoDataset format (MP4 + .txt captions).

Camera view: primary_images_jpeg (third-person/agentview) only, matching the mimic-video paper
which uses a single-view backbone. Wrist camera is intentionally excluded.

Codec: H.264 lossless (crf=0) via ffmpeg to avoid adding further compression artifacts on top
of the JPEG-compressed source frames already stored in the HDF5.

Usage:
    uv run --with h5py --with pillow --with tqdm python scripts/prepare_libero_cosmos_dataset.py \
        --src datasets/libero_cosmos/all_episodes \
        --out datasets/libero_cosmos_mp4 \
        [--fps 10] \
        [--val-fraction 0.1]

    # Include only successful episodes:
        [--success-only]

Output structure:
    datasets/libero_cosmos_mp4/
        train/
            videos/   <episode_name>.mp4   (H.264 lossless)
            metas/    <episode_name>.txt   (task_description caption)
        val/
            videos/   <episode_name>.mp4
            metas/    <episode_name>.txt

num_frames for training: 93
    Aligned with agibot config (state_t=24, temporal_compression_factor=4):
    formula: (state_t - 1) * temporal_compression + 1 = (24-1)*4+1 = 93
    Episodes with fewer than 94 frames are skipped (minimum needed for random sampling).
"""

import argparse
import io
import random
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

MIN_FRAMES = 94  # num_frames=93 requires total_frames >= 94 for randint(0, max_start_idx) to work


def convert_episode(hdf5_path: Path, out_dir: Path, fps: int, success_only: bool) -> bool:
    with h5py.File(hdf5_path, "r") as f:
        success = bool(f.attrs.get("success", True))
        if success_only and not success:
            return False

        task_description = f.attrs.get("task_description", "robot manipulation task")
        raw_frames = f["primary_images_jpeg"]  # shape (T,) of JPEG bytes, agentview camera

        frames = []
        for i in range(len(raw_frames)):
            img = np.array(Image.open(io.BytesIO(raw_frames[i].tobytes())))  # RGB uint8 (256,256,3)
            frames.append(img)

    if len(frames) < MIN_FRAMES:
        return False

    H, W = frames[0].shape[:2]  # 256x256
    name = hdf5_path.stem
    video_path = out_dir / "videos" / f"{name}.mp4"

    # Write frames as raw RGB into a temp file, then encode with ffmpeg H.264 lossless (crf=0).
    # This avoids additional lossy compression on top of the already JPEG-compressed source frames.
    with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as tmp:
        tmp_path = tmp.name
        for frame in frames:
            tmp.write(frame.tobytes())  # RGB bytes

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}",
            "-framerate", str(fps),
            "-i", tmp_path,
            "-c:v", "libx264",
            "-crf", "0",          # lossless H.264
            "-preset", "ultrafast",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    Path(tmp_path).unlink()

    (out_dir / "metas" / f"{name}.txt").write_text(str(task_description))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="datasets/libero_cosmos/all_episodes")
    parser.add_argument("--out", type=str, default="datasets/libero_cosmos_mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of episodes for validation")
    parser.add_argument("--success-only", action="store_true", help="Skip failed episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    for split in ("train", "val"):
        (out_dir / split / "videos").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "metas").mkdir(exist_ok=True)

    hdf5_files = sorted(src_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files in {src_dir}")

    # Reproducible train/val split
    rng = random.Random(args.seed)
    shuffled = list(hdf5_files)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_set = set(p.name for p in shuffled[:n_val])

    converted_train, converted_val, skipped = 0, 0, 0
    for hdf5_path in tqdm(hdf5_files):
        split = "val" if hdf5_path.name in val_set else "train"
        ok = convert_episode(hdf5_path, out_dir / split, args.fps, args.success_only)
        if ok:
            if split == "val":
                converted_val += 1
            else:
                converted_train += 1
        else:
            skipped += 1

    print(f"\nDone. Train: {converted_train}, Val: {converted_val}, Skipped: {skipped}")
    print(f"Output: {out_dir}")
    print(f"\nVideoDataset config:")
    print(f"  dataset_dir = '{out_dir}/train'  (train)")
    print(f"  dataset_dir = '{out_dir}/val'    (val)")
    print(f"  video_size  = (256, 256)")


if __name__ == "__main__":
    main()
