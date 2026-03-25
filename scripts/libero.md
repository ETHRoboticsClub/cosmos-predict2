# LIBERO LoRA fine-tuning

LoRA fine-tuning of Cosmos Predict2 2B (480p, 10fps) on nvidia/LIBERO-Cosmos-Policy.
Based on [mimic-video](https://arxiv.org/abs/2512.15692) Table IV hyperparameters.

## AWS setup (recommended)

**Instance:** `p4de.24xlarge`
- 8× NVIDIA A100 **80 GB** HBM2e, 96 vCPUs, 1.1 TB RAM
- Preferred over `p4d.24xlarge` (40 GB A100s) for headroom with FSDP + context parallelism
- run takes ~4–8 h 

**AMI:** *Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)*
- Search in EC2 console: AMIs → Community AMIs → filter `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260222`
- Comes with CUDA 12.x, cuDNN, NCCL pre-installed; matches the `cu126` extras in `pyproject.toml`
- Select the latest dated build for your region

**Storage:** attach at least 200 GB gp3 EBS (repo + checkpoints + dataset + MP4s)

**Spot vs. on-demand:** p4de spot can be ~60% cheaper but interruptions are common for long runs;
use on-demand or set up checkpoint-resume before using spot.

---

1. clone repo
```
git clone -b libero https://github.com/ETHRoboticsClub/cosmos-predict2.git
```

2. install
```
uv sync --extra cu126
source .venv/bin/activate
```

3. download model Cosmos-Predict2-2B-Video2World
```
uv tool install -U "huggingface_hub[cli]"
hf auth login
# accept terms: https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/tree/main
# can ignore Llama Guardrail errors

python scripts/download_checkpoints.py --model_types video2world --model_sizes 2B
```

4. download libero dataset (~27 GB)
```
huggingface-cli download nvidia/LIBERO-Cosmos-Policy \
  --repo-type dataset --include "all_episodes/*" \
  --local-dir datasets/libero_cosmos
```

5. convert HDF5 → MP4 + captions, split train/val (10%)
```
uv run --with h5py --with pillow --with tqdm \
  python scripts/prepare_libero_cosmos_dataset.py \
  --src datasets/libero_cosmos/all_episodes \
  --out datasets/libero_cosmos_mp4 \
  --fps 10
```
output: `datasets/libero_cosmos_mp4/train/` and `datasets/libero_cosmos_mp4/val/`
episodes with fewer than 94 frames are skipped

6. generate T5 embeddings (run for both splits)
```
python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/train
python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/val
```

7. (optional) set up W&B logging
```
pip install wandb
wandb login   # paste API key from wandb.ai/settings
export WANDB_API_KEY=<your_key>
export WANDB_PROJECT=cosmos-libero   # project name in W&B
```
training will log `train/loss` and `clip_grad_norm/{image,video}` to W&B.
omit `WANDB_API_KEY` to skip W&B entirely (stdout only).

8. train (8× GPU, p4de.24xlarge recommended)
```
IMAGINAIRE_OUTPUT_ROOT=outputs bash scripts/train_libero_cosmos.sh
```
checkpoints saved every 500 steps to:
`outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints/`

9. convert checkpoint for inference
```
CKPT_DIR=outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints
ITER=$(cat $CKPT_DIR/latest_checkpoint.txt)
python convert_distcp.py $CKPT_DIR/$ITER/model $CKPT_DIR/$ITER
```
produces `model_ema_bf16.pt`

9. evaluate (base vs. fine-tuned, 5 val episodes)
```
python scripts/eval_libero_cosmos.py --out eval/base
python scripts/eval_libero_cosmos.py --out eval/finetuned \
  --lora-checkpoint $CKPT_DIR/$ITER/model_ema_bf16.pt
```
outputs `*_comparison.mp4` (ground truth | generated) for each episode