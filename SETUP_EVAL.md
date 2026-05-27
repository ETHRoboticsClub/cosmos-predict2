# Instance Setup

| Setting | Value |
|---------|-------|
| Instance type | `g7e.8xlarge` |
| Platform | Linux/UNIX |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04) 20260517 |

---

## EC2 Terminal Commands

### 1. Clone repo and switch branch

```bash
git clone https://github.com/ETHRoboticsClub/cosmos-predict2.git
cd cosmos-predict2
git checkout mateo-raph/cosmos-eval
```

### 2. Install uv and create environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv lock && uv sync --extra cu128
```

### 3. Check environment

```bash
uv run python scripts/test_environment.py
```

### 4. HuggingFace login

Accept the [model repo conditions](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) in your browser first, then:

```bash
uv run huggingface-cli login
```

Your access token needs public repo access.

### 5. Download model tokenizer

```bash
uv run huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
  --include "tokenizer/*" \
  --local-dir checkpoints/nvidia/Cosmos-Predict2-2B-Video2World
```

### 6. Download T5 text encoder

```bash
uv run huggingface-cli download google-t5/t5-11b \
  --local-dir checkpoints/google-t5/t5-11b \
  --exclude "tf_model.h5"
```

### 7. Download checkpoint from S3

Make sure it's the fused version, not just LoRA weights.

```bash
aws login --remote
aws s3 cp s3://ethrc-ml-data-916780037007/robot-learning/checkpoints/cosmos/ . --recursive
```

### 8. Install ffmpeg

```bash
sudo apt update && sudo apt install -y ffmpeg
```

### 9. Run inference

```bash
uv run python scripts/run_lerobot_inference.py \
    --repo_id ETHRC/yams-closed-carton-box-to-migros-basket-go2 \
    --episode 0 \
    --dit_path checkpoints/Cosmos-Predict2-2B-Video2World/model.pt
```

Generated videos are written to `output/` by default (e.g. `generated_*.mp4`).

### 10. Upload results to S3

Use `scripts/upload_eval_videos_to_s3.sh` to upload every `.mp4` in the output folder. Basenames are preserved under the S3 prefix.

```bash
# Default prefix: s3://ethrc-ml-data-916780037007/robot-learning/cosmos-2-eval/
./scripts/upload_eval_videos_to_s3.sh output

# Optional: upload into a run-specific subfolder
./scripts/upload_eval_videos_to_s3.sh output \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-2-eval/yams-ep0/

# Preview commands without uploading
DRY_RUN=1 ./scripts/upload_eval_videos_to_s3.sh output
```

Requires AWS credentials (e.g. `aws login --remote` from step 7). For a single file instead of a folder, use `aws s3 cp` directly:

```bash
aws s3 cp output/generated_0s.mp4 \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-2-eval/generated_0s.mp4
```
