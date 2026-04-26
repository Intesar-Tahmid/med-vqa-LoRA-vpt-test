# [Evaluating Large Vision Language Models on Bangla Medical VQA](https://ahmedrafid023.github.io/med-vqa-page/)

**Accepted at ACL 2026 (A\* Conference)**  
**Project Page:** [ahmedrafid023.github.io/med-vqa-page/](https://ahmedrafid023.github.io/med-vqa-page/) | **Dataset:** [BanglaMedVQA (Hugging Face)](https://huggingface.co/datasets/iiCEMAN/BanglaMedVQA)

---

## Overview

This repository extends the BanglaMedVQA baseline with **Hierarchical Visual Prompt Tuning (VPT)** on top of LoRA fine-tuning. The core idea: LoRA adapts the *language model* weights, but the grounding failures in Condition (18.5%) and Position (31.25%) categories live in the *visual encoder*. VPT adapts the frozen ViT directly via learnable token pools routed by question category.

**What was added on top of the original LoRA baseline:**
- `src/vpt/` — VPT prompt learner, category router, model wrapper
- `train_vpt.py` — VPT training script with two-phase training and debug mode
- `evaluate.py` — per-category Accuracy / BERTScore / LAVE evaluation
- `configs/vpt/` — ablation configuration presets (A1–A5)
- **Fix:** `train_model.py` now writes the `category` column in predictions CSV (required by `evaluate.py`)

---

## Repository Structure

```
med-vqa-lora/
├── src/
│   └── vpt/
│       ├── __init__.py
│       ├── prompt_learner.py     # VPT token pools + Deep VPT hook injection
│       ├── category_router.py    # 2-layer MLP router (question → pool weight)
│       └── vpt_model.py          # Wraps VLM with VPT + LoRA
├── configs/
│   ├── qwen2.5-vl-7b.yaml        # Baseline B0 training (LoRA only)
│   ├── qwen2.5-vl-7b-test.yaml   # Baseline B0 inference
│   ├── med-gemma-4b.yaml         # MedGemma LoRA training
│   ├── med-gemma-4b-test.yaml    # MedGemma LoRA inference
│   └── vpt/
│       ├── A1_shared_no_router.yaml
│       ├── A2_shared_with_router.yaml
│       ├── A3_split_with_router.yaml    # Core VPT contribution
│       ├── A4_split_router_lora.yaml    # Proposed full method
│       └── A5_oracle_router.yaml        # Upper bound
├── data/
│   ├── dataset_info.json
│   ├── chest_x-ray/
│   │   ├── train/   chest_x-ray.csv · chest_x-ray_dataset.json · images/
│   │   ├── val/     chest_x-ray.csv · chest_x-ray_dataset.json (see §Data Split)
│   │   └── test/    chest_x-ray.csv · chest_x-ray_dataset.json · images/
│   └── medicat/
│       └── train/   medicat.csv · medicat_dataset.json · images/
├── output/                  # Model outputs, checkpoints, predictions
├── results/                 # Per-ablation evaluation JSON files
├── prepare_data.py          # CSV → LLaMA Factory ShareGPT JSON
├── train_model.py           # Baseline B0: LLaMA Factory LoRA training + inference
├── train_vpt.py             # VPT training (all ablations A1–A5)
├── evaluate.py              # Per-category Accuracy / BERTScore / LAVE
├── requirements.txt         # Core dependencies
├── requirements_vpt.txt     # VPT-specific additions (sentence-transformers, bert-score, openai)
└── Dockerfile               # Container for local/Docker use
```

---

## Setup

### Lightning.ai (recommended for training)

Lightning.ai provides A100 80GB GPUs. Open a Studio terminal and run:

```bash
# Clone and enter the repo
git clone <your-repo-url>
cd med-vqa-lora

# Install all dependencies
pip install -r requirements.txt -r requirements_vpt.txt

# Set OpenAI key for LAVE evaluation (optional)
export OPENAI_API_KEY=your_key_here
```

> **Lightning.ai tip:** persist your pip installs by adding them to the Studio's startup script, or use a `requirements.txt` attached to the Studio so packages survive restarts.

### Docker (local / cloud)

```bash
docker build -t med-vqa:latest .
docker run --gpus all --ipc=host --rm --env-file .env \
    -v $(pwd):/app med-vqa:latest bash
```

### Local (no GPU — debug only)

```bash
pip install -r requirements.txt -r requirements_vpt.txt
```

---

## Step 0: Prepare Data

```bash
# Prepare all datasets (chest_x-ray train/test + medicat train)
python prepare_data.py --all

# Or prepare a specific split
python prepare_data.py --dataset chest_x-ray --split train
python prepare_data.py --dataset chest_x-ray --split test
```

### Validation Split (required for VPT experiments)

The VPT training script monitors router accuracy on a validation set during training. Create a stratified 10% validation split from the training data:

```bash
python - <<'EOF'
import pandas as pd
from sklearn.model_selection import train_test_split
import os, json

df = pd.read_csv("data/chest_x-ray/train/chest_x-ray.csv")
train_df, val_df = train_test_split(
    df, test_size=0.1, stratify=df["category"], random_state=42
)

os.makedirs("data/chest_x-ray/val", exist_ok=True)
val_df.to_csv("data/chest_x-ray/val/chest_x-ray.csv", index=False)
train_df.to_csv("data/chest_x-ray/train/chest_x-ray.csv", index=False)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")
print(val_df["category"].value_counts())
EOF

# Rebuild JSON after updating CSV
python prepare_data.py --dataset chest_x-ray --split train
python prepare_data.py --dataset chest_x-ray --split val \
    --csv data/chest_x-ray/val/chest_x-ray.csv \
    --images data/chest_x-ray/train/images \
    --output data/chest_x-ray/val/chest_x-ray_dataset.json
```

---

## Step 1: Reproduce Baseline B0 (LoRA only)

Verify your setup matches the paper's Table 4 results before running VPT experiments.

```bash
# Train
python train_model.py --config configs/qwen2.5-vl-7b.yaml

# Inference (generates output/chest_x-ray_test_predictions.csv)
python train_model.py --config configs/qwen2.5-vl-7b-test.yaml

# Evaluate
python evaluate.py \
    --predictions_csv output/chest_x-ray_test_predictions.csv \
    --output_json results/B0_results.json
```

Expected B0 results (Table 4 of the paper):

| Category    | Acc   | LAVE  |
|-------------|-------|-------|
| Modality    | 96.0% | ~96%  |
| Organ       | 83.8% | ~84%  |
| Abnormality | 75.0% | ~75%  |
| Condition   | 18.5% | ~18%  |
| Position    | 31.3% | ~31%  |
| **Overall** | **49.3%** | **~49%** |

If your numbers are within ±1%, proceed to VPT experiments.

---

## Step 2–6: VPT Ablation Experiments

Run all five ablations in order. Each one adds exactly one component, isolating its contribution.

### A1 — Shared VPT, no router, no LoRA

```bash
python train_vpt.py \
    --output_dir output/A1 \
    --no-add_lora \
    --num_gen_tokens 20 \
    --num_spec_tokens 20
```

### A2 — Shared VPT + router, no LoRA

```bash
python train_vpt.py \
    --output_dir output/A2 \
    --no-add_lora \
    --num_gen_tokens 20 \
    --num_spec_tokens 20
```

### A3 — Split pools + router, no LoRA (core VPT contribution)

```bash
python train_vpt.py \
    --output_dir output/A3 \
    --no-add_lora
```

### A4 — Split + router + LoRA (proposed full method)

```bash
python train_vpt.py \
    --output_dir output/A4 \
    --add_lora \
    --gradient_checkpointing
```

### A5 — Oracle router upper bound

```bash
python train_vpt.py \
    --output_dir output/A5 \
    --add_lora \
    --use_oracle_router \
    --gradient_checkpointing
```

> A5 uses ground-truth categories at training time. If A4 approaches A5, the learned router is near-optimal.

---

## Step 7: Evaluate All Conditions

```bash
for COND in B0 A1 A2 A3 A4 A5; do
    python evaluate.py \
        --predictions_csv output/${COND}/chest_x-ray_test_predictions.csv \
        --output_json results/${COND}_results.json
done
```

To skip LAVE (no OpenAI key required):

```bash
python evaluate.py --predictions_csv output/A4/... --no_lave
```

---

## Step 8: MedGemma Backbone

Repeat A4 with MedGemma to validate generalisability:

```bash
python train_vpt.py \
    --model_name google/medgemma-4b-it \
    --output_dir output/A4_medgemma \
    --add_lora \
    --gradient_checkpointing
```

---

## Debug Mode (CPU smoke-test)

Verifies the training loop and gradient flow without a GPU. **Not for benchmarking.**

```bash
# Quick smoke-test (10 samples, 1 epoch, CPU, no LoRA, 2 prompt tokens)
python train_vpt.py \
    --debug \
    --train_json data/chest_x-ray/train/chest_x-ray_dataset.json \
    --train_csv  data/chest_x-ray/train/chest_x-ray.csv \
    --output_dir output/debug

# Use a smaller model to save RAM/time on CPU
python train_vpt.py \
    --debug \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --debug_samples 5 \
    --output_dir output/debug
```

Debug mode automatically sets:
- `device = cpu`, `num_epochs = 1`, `batch_size = 1`
- `num_gen_tokens = 2`, `num_spec_tokens = 2`
- `add_lora = False`, `gradient_checkpointing = False`
- `log_every = 1` (prints loss every step)

Check that loss decreases over the 10 steps — this confirms VPT gradients are flowing.

> **Note:** Loading a 7B model on CPU is very slow (~15 min). Use the 3B model for faster iteration. For actual quality comparison against LoRA, you need an A100 (Lightning.ai).

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_gen_tokens` | 16 per layer | Generalized pool (Modality, Organ) |
| `num_spec_tokens` | 24 per layer | Specialized pool (Abnormality, Condition, Position) |
| `lora_rank` | 16 | Identical to paper baseline |
| `lora_alpha` | 32 | Identical to paper baseline |
| `lora_dropout` | 0.05 | Slight regularisation |
| `lr_vpt` | 2e-4 | Phase 1; reduced to 1e-4 in Phase 2 via cosine decay |
| `lr_router` | 1e-3 | Higher — MLP converges quickly |
| `lr_lora` | 1e-4 | Same as paper |
| `batch_size` | 4 (+ `grad_accum=4` → eff. 16) | A100 80GB |
| `num_epochs` | 3 (1 warmup + 2 joint) | Phase 1 = epoch 1 only |
| `max_seq_len` | 512 | Identical to paper |
| `lambda_router` | 0.01 | Entropy regularisation weight |
| `mu_cat` | 0.1 | Category supervision weight |
| Sentence encoder | `paraphrase-multilingual-mpnet-base-v2` | 768-d, handles Bangla |

---

## Ablation Summary

| ID | Name | VPT Pool | Router | LoRA |
|----|------|----------|--------|------|
| B0 | Paper LoRA baseline | — | — | ✓ |
| A1 | Shared VPT, no router | Shared (equal blend) | — | — |
| A2 | Shared VPT + router | Shared | ✓ | — |
| A3 | Split pools + router | Gen + Spec | ✓ | — |
| **A4** | **Proposed method** | **Gen + Spec** | **✓** | **✓** |
| A5 | Oracle router (upper bound) | Gen + Spec | Oracle | ✓ |

### Expected Targets (A4 vs B0)

| Category | B0 (paper) | A4 target | Delta |
|----------|-----------|-----------|-------|
| Modality | 96.0% | ~97% | +1 pt |
| Organ | 83.8% | ~87% | +3 pt |
| Abnormality | 75.0% | ~83% | +8 pt |
| **Condition** | **18.5%** | **~38%** | **+20 pt** |
| **Position** | **31.3%** | **~50%** | **+18 pt** |
| **Overall** | **49.3%** | **~62%** | **+13 pt** |

---

## What to Record Per Run

For each ablation, save:
- `results/A*_results.json` — per-category Acc / BScore / LAVE
- `output/A*/train_loss.csv` — loss curve (auto-generated by `train_vpt.py`)
- Router accuracy logged to console at end of each epoch (val set)
- GPU memory and time per epoch (Lightning.ai dashboard)
- 5+ qualitative examples where A4 is correct but B0 is wrong (for paper analysis section)

---

## Implementation Notes

### Deep VPT Design

The VPT implementation uses the **Deep VPT** approach (tokens injected at every layer):

1. **Pre-hook** on each ViT block strips the previous layer's prompt tokens, then prepends fresh tokens for the current layer — keeping sequence length constant throughout the ViT.
2. **Post-hook** on the final ViT block strips prompt tokens before they reach the visual merger (`visual_merger` in Qwen2.5-VL). This ensures the merger receives the original number of patch tokens.
3. Always `max(num_gen_tokens, num_spec_tokens)` tokens per layer — consistent across all ablations.

### Inspecting Your ViT Structure

If auto-detection fails, inspect the model manually:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
for name, _ in model.named_modules():
    if "block" in name or "layer" in name:
        print(name)
# Look for repeating patterns like visual.blocks.0, visual.blocks.1, ...
# Then pass --num_vit_layers <count> explicitly
```

### Gradient Checkpointing

Recommended for A4/A5 on A100 80GB (both VPT tokens and LoRA active simultaneously):

```bash
python train_vpt.py --output_dir output/A4 --add_lora --gradient_checkpointing
```

---

## Output Files

### Baseline (train_model.py)
- `output/adapter_model.safetensors` — LoRA weights
- `output/chest_x-ray_test_predictions.csv` — predictions with `category` column populated

### VPT (train_vpt.py)
- `output/A4/checkpoint-epoch1/vpt_router.pt` — VPT token pools + router MLP state
- `output/A4/checkpoint-epoch1/adapter_model.safetensors` — LoRA weights (if `--add_lora`)
- `output/A4/train_loss.csv` — per-step loss curve

### Evaluation
- `results/A4_results.json` — `{"accuracy": {...}, "bertscore": {...}, "lave": {...}}`

---

## Configuration Files

Config files in `configs/vpt/` document parameter presets for each ablation but are not loaded automatically by `train_vpt.py`. Use them as a reference for the exact flags to pass. All hyperparameters can also be overridden directly on the command line.

---

## Citation

```bibtex
@inproceedings{banglamedvqa2026,
  title     = {Evaluating Large Vision Language Models on Bangla Medical VQA},
  booktitle = {ACL 2026},
  url       = {https://ahmedrafid023.github.io/med-vqa-page/}
}
```
