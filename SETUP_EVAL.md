#### INSTANCE SETUP ######

Instance type
g7e.2xlarge

Platform details
Linux/UNIX

AMI name
Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04) 20260517

#### EC2 TERMINAL COMMANDS #######

1. Clone repo : https://github.com/ETHRoboticsClub/cosmos-predict2

2. Go on branch: git checkout mateo-raph/cosmos-eval

2. Download uv: curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

2. Create env : uv lock && uv sync --extra cu126

3. Check environment is ok: uv run python scripts/test_environment.py

3. Huggingface login -> access token should have public repo access (you have to accept the repo conditions in browser first): uv run huggingface-cli login

4. Download model (tokenizer) : uv run huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
  --include "tokenizer/*" \
  --local-dir checkpoints/nvidia/Cosmos-Predict2-2B-Video2World

5. Login to AWS: aws login --remote

5. Download checkpoint you want (make sure it's not just LoRA weights, but the fused version): aws s3 cp s3://ethrc-ml-data-916780037007/robot-learning/checkpoints/cosmos/ . --recursive

5. Download t5 only: uv run huggingface-cli download google-t5/t5-11b \
  --local-dir checkpoints/google-t5/t5-11b \
  --exclude "tf_model.h5"

6. Download ffmpeg: sudo apt update
sudo apt install -y ffmpeg

6. Run inference:  python scripts/run_lerobot_inference.py \
    --repo_id ETHRC/yams-closed-carton-box-to-migros-basket-go2 \
    --episode 0 \
    --dit_path checkpoints/Cosmos-Predict2-2B-Video2World/model.pt
