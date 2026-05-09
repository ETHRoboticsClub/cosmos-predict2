# LeRobot → Cosmos Predict2 fine-tuning

LoRA fine-tuning of Cosmos Predict2 2B (480p, 10fps) on a LeRobot v3.0 dataset.
Based on [mimic-video](https://arxiv.org/abs/2512.15692) Table IV hyperparameters.

This guide uses `ETHRC/yams-closed-carton-box-to-migros-basket-go2` as the example dataset.
Replace the HuggingFace repo ID and output directory names for your own dataset.

---

## Format overview

LeRobot v3.0 datasets differ from Libero HDF5 in three ways relevant to this pipeline:

| | Libero | LeRobot v3.0 |
|---|---|---|
| Source format | HDF5 with JPEG frames | Parquet + MP4 (one file may contain multiple episodes) |
| Captions | `hdf5.attrs["task_description"]` | `meta/tasks.parquet` |
| FPS | 10 fps natively | Typically 30 fps → must downsample to 10 fps |

FPS downsampling is done with ffmpeg's `fps=10` filter (frame dropping, every 3rd frame kept —
no interpolation, no slow-motion). The MIN_FRAMES check (94 output frames) is applied *after*
downsampling, so a source episode needs at least `94 × (src_fps / 10)` frames (e.g. 282 at 30 fps).

---

## Steps

### 1. Install

```
git clone -b libero https://github.com/ETHRoboticsClub/cosmos-predict2.git
cd cosmos-predict2
uv sync --extra cu126
source .venv/bin/activate
```

### 2. Download model checkpoint

```
uv tool install -U "huggingface_hub[cli]"
hf auth login
# accept terms: https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/tree/main

python scripts/download_checkpoints.py \
  --model_types video2world \
  --model_sizes 2B \
  --resolution 480 \
  --fps 10
```

### 3. Download the LeRobot dataset

```
huggingface-cli download ETHRC/yams-closed-carton-box-to-migros-basket-go2 \
  --repo-type dataset \
  --local-dir datasets/yams_lerobot
```

### 4. Convert to VideoDataset format

Converts LeRobot v3.0 → `{train,val}/{videos,metas}` (MP4 + .txt captions).

- Extracts the **topdown** camera view (`observation.images.topdown`)
- Downsamples from **30 fps → 10 fps** (keeps every 3rd frame)
- Scales to **480×640** (H×W, both divisible by 16)
- Splits episodes into train (90%) / val (10%)

```
uv run --with pyarrow --with pandas --with tqdm \
  python scripts/prepare_lerobot_cosmos_dataset.py \
  --src datasets/yams_lerobot \
  --out datasets/yams_cosmos_mp4 \
  --camera top \
  --fps 10 \
  --video-size 480 640
```

output: `datasets/yams_cosmos_mp4/train/` and `datasets/yams_cosmos_mp4/val/`
episodes with fewer than 94 output frames after downsampling are skipped.

To use a different camera (e.g. wrist), pass `--camera wrist`. The script matches
by substring against the dataset's video keys.

### 5. Generate T5 embeddings

```
python -m scripts.get_t5_embeddings --dataset_path datasets/yams_cosmos_mp4/train
python -m scripts.get_t5_embeddings --dataset_path datasets/yams_cosmos_mp4/val
```

### 6. (Optional) W&B logging

```
pip install wandb
export WANDB_API_KEY=<your_key>
export WANDB_PROJECT=cosmos-yams
```

### 7. Smoke test

```
IMAGINAIRE_OUTPUT_ROOT=outputs torchrun \
  --nproc_per_node=8 \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_yams \
  trainer.max_iter=5 \
  trainer.validation_iter=1 \
  trainer.max_val_iter=2 \
  checkpoint.save_iter=999999
```

### 8. Train (8× GPU, p4de.24xlarge / A100 80 GB)

```
NPROC=8 IMAGINAIRE_OUTPUT_ROOT=outputs uv run torchrun \
  --nproc_per_node="${NPROC:-1}" \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_yams \
  model_parallel.context_parallel_size=1 \
  dataloader_train.batch_size=8 \
  dataloader_val.batch_size=2 \
  trainer.max_iter=7000 \
  trainer.grad_accum_iter=2 \
  trainer.validation_iter=50 \
  trainer.max_val_iter=10 \
  trainer.callbacks.draw_sample.every_n=200 \
  trainer.callbacks.draw_sample.is_sample=True \
  trainer.callbacks.draw_sample.show_all_frames=True \
  "trainer.callbacks.draw_sample.guidance=[7.0]" \
  checkpoint.save_iter=500
```

effective batch=128 (batch_size=8 × 8 DP ranks × grad_accum=2), lr=1.778e-4

---

## Experiment config

Add a config file at `cosmos_predict2/configs/base/experiment/yams.py` modelled on
`libero_cosmos.py`, pointing `dataset_dir` at the prepared MP4 directories and setting
`video_size=(480, 640)`.

```python
yams_dataset_train = L(Dataset)(
    dataset_dir="datasets/yams_cosmos_mp4/train",
    num_frames=93,
    video_size=(480, 640),
)
yams_dataset_val = L(Dataset)(
    dataset_dir="datasets/yams_cosmos_mp4/val",
    num_frames=93,
    video_size=(480, 640),
)
```

---

## Notes

- **Resolution**: 480×640 matches the source camera resolution exactly, so no upscaling is needed.
  Both dimensions are divisible by 16 (8× VAE spatial compression × 2× DiT patch size).
- **Small datasets**: with only a handful of episodes, overfitting is expected. Use the val loss
  curve and `draw_sample` visualisations to decide when to stop.
- **Multiple cameras**: the script selects one camera per run. Re-run with `--camera wrist` and
  a different `--out` to prepare a second view if needed.
