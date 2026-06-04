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

import os
import time
import traceback
import warnings
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo
from cosmos_predict2.data.embedding_cache import get_embedding_cache_key, load_t5_embedding_cached
from imaginaire.utils import log

"""
Test the dataset with the following command:
python -m cosmos_predict2.data.dataset_video
"""


class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        video_size,
        exclude_video_indices=None,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames

        video_dir = os.path.join(self.dataset_dir, "videos")
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        self.all_video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.all_video_paths = sorted(self.all_video_paths)
        # remove video paths that does not have t5_embedding
        self.all_video_paths = [
            path
            for path in self.all_video_paths
            if os.path.exists(os.path.join(self.t5_dir, os.path.basename(path).replace(".mp4", ".pickle")))
        ]
        excluded_indices = self._normalize_video_indices(exclude_video_indices, len(self.all_video_paths))
        self.video_paths = [path for idx, path in enumerate(self.all_video_paths) if idx not in excluded_indices]
        log.info(
            f"{len(self.video_paths)} videos in total "
            f"({len(excluded_indices)} excluded from {len(self.all_video_paths)} available)"
        )

        self.wrong_number = 0
        self._t5_embedding_cache = {}
        self._last_load_profile = {}
        self._slow_load_warn_s = float(os.getenv("COSMOS_SLOW_LOAD_WARN_S", "0.5"))
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size), mode="center_crop")])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    @staticmethod
    def _normalize_video_indices(indices, total: int) -> set[int]:
        if indices is None:
            return set()
        if isinstance(indices, int):
            indices = [indices]
        normalized = set()
        for index in indices:
            index = int(index)
            if index < 0:
                index += total
            if index < 0 or index >= total:
                raise IndexError(f"Video index {index} is out of range for {total} videos")
            normalized.add(index)
        return normalized

    def _load_video(self, video_path, start_frame: int | None = None) -> tuple[np.ndarray, float]:
        t_open0 = time.perf_counter()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        t_open = time.perf_counter() - t_open0

        t_len0 = time.perf_counter()
        total_frames = len(vr)
        t_len = time.perf_counter() - t_len0
        if total_frames < self.sequence_length:
            # If there are not enough frames, let it fail
            warnings.warn(  # noqa: B028
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )
            raise ValueError(f"Video {video_path} has insufficient frames.")

        max_start_idx = total_frames - self.sequence_length
        if start_frame is None:
            # randomly sample a sequence of frames
            start_frame = np.random.randint(0, max_start_idx + 1)
        else:
            start_frame = int(start_frame)
            if start_frame < 0 or start_frame > max_start_idx:
                raise ValueError(
                    f"Requested start_frame={start_frame} for {video_path}, "
                    f"but valid range is [0, {max_start_idx}]"
                )
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        t_decode0 = time.perf_counter()
        frame_data = vr.get_batch(frame_ids).asnumpy()
        t_decode = time.perf_counter() - t_decode0

        t_seek0 = time.perf_counter()
        vr.seek(0)  # set video reader point back to 0 to clean up cache
        t_seek = time.perf_counter() - t_seek0

        t_fps0 = time.perf_counter()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        t_fps = time.perf_counter() - t_fps0

        self._last_load_profile = {
            "open_s": t_open,
            "len_s": t_len,
            "decode_s": t_decode,
            "seek_s": t_seek,
            "fps_s": t_fps,
            "total_frames": total_frames,
            "start_frame": start_frame,
        }
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps

    def _get_frames(self, video_path: str, start_frame: int | None = None) -> tuple[torch.Tensor, float]:
        t_load0 = time.perf_counter()
        frames, fps = self._load_video(video_path, start_frame=start_frame)
        t_load = time.perf_counter() - t_load0

        t_cast0 = time.perf_counter()
        frames = frames.astype(np.uint8)
        t_cast = time.perf_counter() - t_cast0

        t_tensor0 = time.perf_counter()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        t_tensor = time.perf_counter() - t_tensor0

        t_resize0 = time.perf_counter()
        frames = self.preprocess(frames)
        t_resize = time.perf_counter() - t_resize0

        t_clamp0 = time.perf_counter()
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        t_clamp = time.perf_counter() - t_clamp0

        self._last_load_profile["load_video_s"] = t_load
        self._last_load_profile["astype_s"] = t_cast
        self._last_load_profile["to_tensor_s"] = t_tensor
        self._last_load_profile["resize_s"] = t_resize
        self._last_load_profile["clamp_s"] = t_clamp
        self._last_load_profile["preprocess_s"] = t_cast + t_tensor + t_resize + t_clamp
        return frames, fps

    def _build_item(self, video_path: str, start_frame: int | None, source_index: int) -> dict | Any:
        data = dict()
        _t0 = time.perf_counter()
        video, fps = self._get_frames(video_path, start_frame=start_frame)
        _load_s = time.perf_counter() - _t0
        if _load_s > self._slow_load_warn_s:
            p = self._last_load_profile
            log.warning(
                f"Slow data load: {video_path} took {_load_s:.2f}s "
                f"(open={p.get('open_s', 0.0):.2f}s len={p.get('len_s', 0.0):.2f}s "
                f"decode={p.get('decode_s', 0.0):.2f}s seek={p.get('seek_s', 0.0):.2f}s "
                f"fps={p.get('fps_s', 0.0):.2f}s load_video={p.get('load_video_s', 0.0):.2f}s "
                f"astype={p.get('astype_s', 0.0):.2f}s to_tensor={p.get('to_tensor_s', 0.0):.2f}s "
                f"resize={p.get('resize_s', 0.0):.2f}s clamp={p.get('clamp_s', 0.0):.2f}s "
                f"preprocess={p.get('preprocess_s', 0.0):.2f}s "
                f"start={p.get('start_frame', -1)} total_frames={p.get('total_frames', -1)})"
            )
        video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
        t5_embedding_path = os.path.join(
            self.t5_dir,
            os.path.basename(video_path).replace(".mp4", ".pickle"),
        )
        data["video"] = video
        data["video_name"] = {
            "video_path": video_path,
            "t5_embedding_path": t5_embedding_path,
            "video_index": source_index,
            "start_frame": -1 if start_frame is None else start_frame,
        }

        _, _, h, w = video.shape

        cache_key = get_embedding_cache_key(t5_embedding_path)
        t5_embedding, t5_text_mask = load_t5_embedding_cached(
            self._t5_embedding_cache, cache_key, t5_embedding_path
        )

        data["t5_text_embeddings"] = t5_embedding
        data["t5_text_mask"] = t5_text_mask
        data["fps"] = fps
        data["image_size"] = torch.tensor([h, w, h, w])
        data["num_frames"] = self.sequence_length
        data["padding_mask"] = torch.zeros(1, h, w)

        return data

    def get_fixed_sample(self, video_index: int, start_frame: int, from_all_videos: bool = True) -> dict | Any:
        paths = self.all_video_paths if from_all_videos else self.video_paths
        if video_index < 0:
            video_index += len(paths)
        if video_index < 0 or video_index >= len(paths):
            raise IndexError(f"Video index {video_index} is out of range for {len(paths)} videos")
        return self._build_item(paths[video_index], start_frame=start_frame, source_index=video_index)

    def __getitem__(self, index) -> dict | Any:
        try:
            return self._build_item(self.video_paths[index], start_frame=None, source_index=index)
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            return self[np.random.randint(len(self.video_paths))]


if __name__ == "__main__":
    dataset = Dataset(
        dataset_dir="datasets/benchmark_train/gr1",
        num_frames=93,
        video_size=[480, 832],
    )

    indices = [0, 13, -1]
    for idx in indices:
        data = dataset[idx]
        log.info(
            f"{idx=} "
            f"{data['video'].sum()=}\n"
            f"{data['video'].shape=}\n"
            f"{data['video_name']=}\n"
            f"{data['t5_text_embeddings'].shape=}\n"
            "---"
        )
