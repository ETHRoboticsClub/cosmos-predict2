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

import argparse
import os
import pickle

from imaginaire.constants import T5_MODEL_DIR

"""example command
python -m scripts.get_t5_embeddings --dataset_path datasets/hdvila
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="datasets/hdvila", help="Root path to the dataset")
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum length of the text embedding",
    )
    parser.add_argument("--cache_dir", type=str, default=T5_MODEL_DIR, help="Directory to cache the T5 model")
    return parser.parse_args()


def main(args) -> None:
    metas_dir = os.path.join(args.dataset_path, "metas")
    if not os.path.isdir(metas_dir):
        raise FileNotFoundError(f"Missing metadata directory: {metas_dir}")

    metas_list = [
        os.path.join(metas_dir, filename) for filename in sorted(os.listdir(metas_dir)) if filename.endswith(".txt")
    ]
    if not metas_list:
        raise RuntimeError(
            f"No prompt metadata found in {metas_dir}. "
            "Run ./commands/preprocess.sh first, or pass --dataset_path to a preprocessed dataset with metas/*.txt files."
        )

    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    import numpy as np

    from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig

    # Initialize T5
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    prompt_to_embedding = {}
    for meta_filename in metas_list:
        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(meta_filename).replace(".txt", ".pickle"))
        with open(meta_filename) as fp:
            prompt = fp.read().strip()

        if os.path.exists(t5_xxl_filename):
            prompt_to_embedding.setdefault(prompt, None)
            continue

        encoded_text = prompt_to_embedding.get(prompt)
        if encoded_text is None:
            # Compute once per unique prompt, then write a real file for every sample.
            encoded_text_raw, mask_bool = encoder.encode_prompts(
                prompt, max_length=args.max_length, return_mask=True
            )  # list of np.ndarray in (len, embed_dim)
            attn_mask = mask_bool.long()
            lengths = attn_mask.sum(dim=1).cpu()

            encoded_text_np = encoded_text_raw.cpu().numpy().astype(np.float16)

            # trim zeros to save space
            encoded_text = [encoded_text_np[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text_np.shape[0])]
            prompt_to_embedding[prompt] = encoded_text

        # Save T5 embeddings as pickle file (no hardlink/symlink).
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)

    print(f"Prepared {len(metas_list)} embedding files from {len(prompt_to_embedding)} unique prompts.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
