# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute one T5 embedding per unique teleop instruction")
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path containing date/episode folders")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Embedding cache directory. Defaults to <dataset_path>/t5_xxl_instruction_cache",
    )
    parser.add_argument("--episode_glob", type=str, default="*/episode_*", help="Glob relative to dataset_path")
    parser.add_argument("--max_length", type=int, help="Maximum length of the text embedding")
    parser.add_argument("--cache_dir", type=str, default=T5_MODEL_DIR, help="Directory containing the T5 model")
    return parser.parse_args()


def instruction_hash(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def collect_instructions(dataset_path: Path, episode_glob: str) -> dict[str, str]:
    instructions: dict[str, str] = {}
    candidate_episode_dirs = [path for path in sorted(dataset_path.glob(episode_glob)) if path.is_dir()]
    if not candidate_episode_dirs:
        candidate_episode_dirs = sorted({path.parent for path in dataset_path.rglob("session_meta.json")})

    for episode_dir in candidate_episode_dirs:
        session_meta_path = episode_dir / "session_meta.json"
        if not session_meta_path.exists():
            continue
        with open(session_meta_path) as fp:
            session_meta = json.load(fp)
        instruction = session_meta.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        instruction = instruction.strip()
        instructions[instruction_hash(instruction)] = instruction
    return instructions


def main(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else dataset_path / "t5_xxl_instruction_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions = collect_instructions(dataset_path, args.episode_glob)
    if not instructions:
        raise FileNotFoundError(f"No instructions found under {dataset_path} with glob {args.episode_glob!r}")

    with open(output_dir / "instruction_index.json", "w") as fp:
        json.dump(instructions, fp, indent=2, sort_keys=True)

    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    for digest, instruction in tqdm(sorted(instructions.items())):
        t5_xxl_filename = output_dir / f"{digest}.pickle"
        if t5_xxl_filename.exists():
            continue

        encoded_text, mask_bool = encoder.encode_prompts(instruction, max_length=args.max_length, return_mask=True)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)

    print(f"Saved {len(instructions)} unique instruction embeddings to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
