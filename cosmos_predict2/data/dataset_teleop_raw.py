# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import pickle
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log


Split = Literal["all", "train", "val"]


@dataclass(frozen=True)
class TeleopEpisode:
    episode_dir: Path
    video_path: Path
    session_meta_path: Path
    instruction: str
    instruction_hash: str


def instruction_hash(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


class TeleopRawDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        *,
        camera_name: str = "camera_top",
        embedding_cache_dir: str | None = None,
        episode_glob: str = "*/episode_*",
        split: Split = "all",
        val_fraction: float = 0.1,
        split_seed: int = 42,
        require_embeddings: bool = True,
    ) -> None:
        """Dataset for raw teleop episodes with one or more camera MP4s.

        Expected episode layout:
            dataset_dir/YYYYMMDD/episode_xxx/
                camera_top-images-rgb.mp4
                session_meta.json  # contains "instruction"

        Text embeddings are cached once per unique instruction at:
            embedding_cache_dir/<sha256(instruction)>.pickle
        """

        super().__init__()
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.sequence_length = num_frames
        self.camera_name = camera_name
        self.embedding_cache_dir = Path(embedding_cache_dir).expanduser() if embedding_cache_dir else (
            self.dataset_dir / "t5_xxl_instruction_cache"
        )
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])
        self._t5_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        episodes = self._discover_episodes(episode_glob)
        self.episodes = self._apply_split(episodes, split, val_fraction, split_seed)
        if require_embeddings:
            self._check_embeddings()
        self.wrong_number = 0

        log.info(
            f"{len(self.episodes)} teleop episodes in {self.dataset_dir} "
            f"(camera={camera_name}, split={split})"
        )

    def __str__(self) -> str:
        return f"{len(self.episodes)} teleop raw samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.episodes)

    def _discover_episodes(self, episode_glob: str) -> list[TeleopEpisode]:
        episodes: list[TeleopEpisode] = []
        video_filename = f"{self.camera_name}-images-rgb.mp4"

        for episode_dir in sorted(self.dataset_dir.glob(episode_glob)):
            if not episode_dir.is_dir():
                continue

            session_meta_path = episode_dir / "session_meta.json"
            video_path = episode_dir / video_filename
            if not session_meta_path.exists() or not video_path.exists():
                continue

            with open(session_meta_path) as fp:
                session_meta = json.load(fp)

            instruction = session_meta.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                warnings.warn(f"Skipping {episode_dir}: session_meta.json has no instruction")  # noqa: B028
                continue

            episodes.append(
                TeleopEpisode(
                    episode_dir=episode_dir,
                    video_path=video_path,
                    session_meta_path=session_meta_path,
                    instruction=instruction.strip(),
                    instruction_hash=instruction_hash(instruction.strip()),
                )
            )

        if not episodes:
            raise FileNotFoundError(
                f"No teleop episodes found under {self.dataset_dir} with glob {episode_glob!r} "
                f"and video {video_filename!r}"
            )
        return episodes

    @staticmethod
    def _apply_split(
        episodes: list[TeleopEpisode],
        split: Split,
        val_fraction: float,
        split_seed: int,
    ) -> list[TeleopEpisode]:
        if split == "all":
            return episodes
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported split {split!r}; expected 'all', 'train', or 'val'")
        if not 0 <= val_fraction < 1:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

        shuffled = list(episodes)
        random.Random(split_seed).shuffle(shuffled)
        n_val = int(len(shuffled) * val_fraction)
        if val_fraction > 0 and n_val == 0:
            n_val = 1
        val_dirs = {episode.episode_dir for episode in shuffled[:n_val]}

        if split == "val":
            return [episode for episode in episodes if episode.episode_dir in val_dirs]
        return [episode for episode in episodes if episode.episode_dir not in val_dirs]

    def _check_embeddings(self) -> None:
        missing: dict[str, str] = {}
        for episode in self.episodes:
            embedding_path = self.embedding_cache_dir / f"{episode.instruction_hash}.pickle"
            if not embedding_path.exists():
                missing[episode.instruction_hash] = episode.instruction

        if missing:
            first_hash, first_instruction = next(iter(missing.items()))
            raise FileNotFoundError(
                f"Missing {len(missing)} unique instruction embeddings under {self.embedding_cache_dir}. "
                f"First missing hash: {first_hash} for instruction {first_instruction!r}. "
                "Run `python -m scripts.get_t5_embeddings_from_teleop_raw --dataset_path ...` first."
            )

    def _load_video(self, video_path: Path) -> tuple[np.ndarray, float]:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            warnings.warn(  # noqa: B028
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )
            raise ValueError(f"Video {video_path} has insufficient frames.")

        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx + 1)
        frame_ids = np.arange(start_frame, start_frame + self.sequence_length).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)
        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 16
        del vr
        return frame_data, fps

    def _get_frames(self, video_path: Path) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def _get_t5_embedding(self, episode: TeleopEpisode) -> tuple[torch.Tensor, torch.Tensor, Path]:
        embedding_path = self.embedding_cache_dir / f"{episode.instruction_hash}.pickle"
        cache_key = str(embedding_path.resolve())
        cached = self._t5_cache.get(cache_key)
        if cached is not None:
            return cached[0], cached[1], embedding_path

        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Missing T5 embedding for instruction {episode.instruction!r}: {embedding_path}. "
                "Run `python -m scripts.get_t5_embeddings_from_teleop_raw --dataset_path ...` first."
            )

        with open(embedding_path, "rb") as f:
            t5_embedding_raw = pickle.load(f)
        assert isinstance(t5_embedding_raw, list)
        assert len(t5_embedding_raw) == 1
        t5_embedding = t5_embedding_raw[0]
        assert isinstance(t5_embedding, np.ndarray)
        assert len(t5_embedding.shape) == 2

        n_tokens = min(t5_embedding.shape[0], CosmosTextEncoderConfig.NUM_TOKENS)
        t5_embedding = t5_embedding[:n_tokens]
        if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
            t5_embedding = np.concatenate(
                [
                    t5_embedding,
                    np.zeros(
                        (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )

        t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
        t5_text_mask[:n_tokens] = 1
        cached_tensors = (torch.from_numpy(t5_embedding), t5_text_mask)
        self._t5_cache[cache_key] = cached_tensors
        return cached_tensors[0], cached_tensors[1], embedding_path

    def __getitem__(self, index) -> dict | Any:
        try:
            episode = self.episodes[index]
            data = dict()
            _t0 = time.perf_counter()
            video, fps = self._get_frames(episode.video_path)
            _load_s = time.perf_counter() - _t0
            if _load_s > 0.5:
                log.warning(f"Slow data load: {episode.video_path} took {_load_s:.2f}s")

            video = video.permute(1, 0, 2, 3)
            _, _, h, w = video.shape

            t5_embedding, t5_text_mask, t5_embedding_path = self._get_t5_embedding(episode)

            data["video"] = video
            data["video_name"] = {
                "episode_dir": str(episode.episode_dir),
                "video_path": str(episode.video_path),
                "session_meta_path": str(episode.session_meta_path),
                "instruction": episode.instruction,
                "instruction_hash": episode.instruction_hash,
                "t5_embedding_path": str(t5_embedding_path),
            }
            data["t5_text_embeddings"] = t5_embedding
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.episodes[index]}. Skipped "
                "(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            return self[np.random.randint(len(self.episodes))]
