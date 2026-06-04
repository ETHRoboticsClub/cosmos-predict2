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

import torch
import torchvision.transforms.functional as F


class Resize_Preprocess:
    def __init__(self, size: tuple[int, int], mode: str = "resize"):
        """
        Initialize the preprocessing class with the target size.
        Args:
        size (tuple): The target height and width as a tuple (height, width).
        """
        self.size = size
        self.mode = mode

    def _resize(self, frame):
        if self.mode == "resize":
            return F.resize(frame, self.size, antialias=True)
        if self.mode != "center_crop":
            raise ValueError(f"Unknown resize mode: {self.mode}")

        target_h, target_w = self.size
        _, input_h, input_w = frame.shape
        scale = max(target_h / input_h, target_w / input_w)
        resize_size = [math.ceil(input_h * scale), math.ceil(input_w * scale)]
        frame = F.resize(frame, resize_size, antialias=True)
        return F.center_crop(frame, self.size)

    def __call__(self, video_frames):
        """
        Apply the transformation to each frame in the video.
        Args:
        video_frames (torch.Tensor): A tensor representing a batch of video frames.
        Returns:
        torch.Tensor: The transformed video frames.
        """
        if video_frames.ndim == 4:
            # Resize each frame in the video
            resized_frames = torch.stack([self._resize(frame) for frame in video_frames])
        else:
            # Resize a single image
            resized_frames = self._resize(video_frames)

        return resized_frames


class ToTensorImage:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, image):
        """
        Args:
            image (torch.tensor, dtype=torch.uint8): Size is (C, H, W)
        Return:
            image (torch.tensor, dtype=torch.float): Size is (C, H, W)
        """
        return to_tensor_image(image)

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


def to_tensor_image(image):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of image tensor
    Args:
        image (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        image (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_image(image)
    if not image.dtype == torch.uint8:
        raise TypeError("image tensor should have data type uint8. Got %s" % str(image.dtype))  # noqa: UP031
    return image.float() / 255.0


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))  # noqa: UP031
    return clip.float() / 255.0


def _is_tensor_image(image: torch.Tensor) -> bool:
    if not torch.is_tensor(image):
        raise TypeError("image should be Tensor. Got %s" % type(image))  # noqa: UP031

    if not image.ndimension() == 3:
        raise ValueError("image should be 3D. Got %dD" % image.dim())  # noqa: UP031

    return True


def _is_tensor_video_clip(clip) -> bool:
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))  # noqa: UP031

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())  # noqa: UP031

    return True
