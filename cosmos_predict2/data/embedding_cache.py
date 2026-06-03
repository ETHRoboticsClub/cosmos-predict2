# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import numpy as np
import torch

from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig


def get_embedding_cache_key(embedding_path: str) -> str:
    meta_path = _embedding_path_to_meta_path(embedding_path)
    if os.path.exists(meta_path):
        with open(meta_path) as fp:
            return f"prompt:{fp.read().strip()}"

    try:
        stat = os.stat(embedding_path)
    except OSError:
        return f"path:{embedding_path}"
    return f"inode:{stat.st_dev}:{stat.st_ino}"


def load_t5_embedding_cached(
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]], cache_key: str, embedding_path: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_key not in cache:
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
        cache[cache_key] = (torch.from_numpy(t5_embedding), t5_text_mask)

    return cache[cache_key]


def _embedding_path_to_meta_path(embedding_path: str) -> str:
    parts = list(os.path.normpath(embedding_path).split(os.sep))
    for index, part in enumerate(parts):
        if part == "t5_xxl":
            parts[index] = "metas"
            break
    meta_path = os.sep.join(parts)
    base, _ = os.path.splitext(meta_path)
    return f"{base}.txt"
