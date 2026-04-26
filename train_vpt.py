#!/usr/bin/env python3
"""
Hierarchical VPT Training Script

Trains VPTPromptLearner + CategoryRouter on top of a frozen ViT encoder,
with optional LoRA on the language model (identical config to the baseline).

Two-phase training:
  Phase 1 (epoch 1): freeze LoRA — warm up VPT tokens and router MLP only
  Phase 2 (epochs 2+): unfreeze LoRA — joint training

Usage (Lightning.ai / A100):
    python train_vpt.py --output_dir output/A4 --add_lora

Usage (CPU debug — verifies the training loop, NOT model quality):
    python train_vpt.py --debug --output_dir output/debug

See configs/vpt/ for per-ablation parameter presets.
"""

import os
import csv
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
import pandas as pd
import yaml
from tqdm import tqdm

from src.vpt.vpt_model import HierarchicalVPTModel
from src.vpt.category_router import CATEGORY_TO_LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BanglaMedVQADataset(Dataset):
    """
    Reads JSON files produced by prepare_data.py and enriches each sample
    with the category label from the original CSV (needed for router supervision).
    """

    def __init__(self, json_path: str, csv_path: str, processor,
                 max_length: int = 512, max_samples: int = None):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if max_samples is not None:
            data = data[:max_samples]
        self.data = data

        df = pd.read_csv(csv_path)
        # Build question_bn → category lookup
        self.cat_lookup: dict = dict(
            zip(df["question_bn"].astype(str), df["category"].astype(str))
        )

        self.processor  = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        entry    = self.data[idx]
        img_path = entry["images"][0]
        question = entry["conversations"][0]["value"].replace("<image>\n", "")
        answer   = entry["conversations"][1]["value"]
        category = self.cat_lookup.get(question, "condition")

        image = Image.open(img_path).convert("RGB")

        # Build the prompt prefix (image + question) WITHOUT the answer,
        # then build the full sequence (prefix + answer) for label masking.
        prefix_msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}]
        prefix_text = self.processor.apply_chat_template(
            prefix_msgs, add_generation_prompt=True
        )
        full_text = prefix_text + answer + self.processor.tokenizer.eos_token

        # Tokenise full sequence with image features
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        # Tokenise the prefix-only text to find where the answer begins.
        # We do NOT include the image again — only need length of text tokens.
        prefix_text_ids = self.processor.tokenizer(
            prefix_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids[0]
        prefix_len = prefix_text_ids.shape[0]

        input_ids      = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        # Build labels: -100 everywhere except over the answer tokens
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        # Mask padding positions
        labels[attention_mask == 0] = -100

        out = {
            "pixel_values":   full_inputs["pixel_values"].squeeze(0),
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "question":       question,
            "category":       category,
        }

        if "image_grid_thw" in full_inputs:
            out["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)

        return out


def collate_fn(batch: list) -> dict:
    """Stack tensors; pass string lists through."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        else:
            result[k] = [b[k] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    lm_loss: torch.Tensor,
    router_alpha: torch.Tensor,
    category_labels: list,
    lambda_router: float = 0.01,
    mu_cat: float = 0.1,
) -> torch.Tensor:
    """
    L_total = L_answer + lambda_router * L_entropy + mu_cat * L_category

    L_entropy regularises the router to avoid collapsing to one pool.
    L_category supervises the router with ground-truth Gen/Spec labels.
    """
    # Router entropy regularisation (maximise → negative sign)
    probs     = torch.stack([router_alpha, 1.0 - router_alpha], dim=-1).clamp(1e-8, 1.0)
    L_entropy = -(probs * probs.log()).sum(dim=-1).mean()
    L_router  = -lambda_router * L_entropy

    # Supervised category loss
    cat_targets = torch.tensor(
        [CATEGORY_TO_LABEL.get(c.lower(), 1) for c in category_labels],
        dtype=router_alpha.dtype,
        device=router_alpha.device,
    )
    L_cat = F.binary_cross_entropy(router_alpha, cat_targets)

    return lm_loss + L_router + mu_cat * L_cat


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_router(model: HierarchicalVPTModel, loader: DataLoader, device: str) -> float:
    """Fraction of samples where the router classifies Gen vs Spec correctly."""
    correct = total = 0
    for batch in loader:
        questions  = batch["question"]
        categories = batch["category"]
        alpha = model.router(questions).to(device)
        preds = (alpha < 0.5).long()   # 0=Gen (alpha≥0.5 wrong), 1=Spec
        # correct if category is Spec and pred=1, or Gen and pred=0
        targets = torch.tensor(
            [CATEGORY_TO_LABEL.get(c.lower(), 1) for c in categories],
            device=device,
        )
        correct += (preds == targets).sum().item()
        total   += len(categories)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = args.device

    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {args.output_dir}")
    if args.debug:
        logger.info("DEBUG MODE — smoke-test only, not for benchmarking")

    # ---- Build model -------------------------------------------------------
    lora_cfg = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05}
    device_map = "auto" if device != "cpu" else {"": "cpu"}

    model = HierarchicalVPTModel(
        model_name      = args.model_name,
        lora_config     = lora_cfg if args.add_lora else None,
        num_gen_tokens  = args.num_gen_tokens,
        num_spec_tokens = args.num_spec_tokens,
        add_lora        = args.add_lora,
        device_map      = device_map,
        num_vit_layers  = args.num_vit_layers,
        disable_router  = args.disable_router,
    )
    model.print_trainable_summary()

    if args.gradient_checkpointing:
        try:
            model.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    # ---- Data --------------------------------------------------------------
    max_samples = args.debug_samples if args.debug else None

    train_dataset = BanglaMedVQADataset(
        json_path   = args.train_json,
        csv_path    = args.train_csv,
        processor   = model.processor,
        max_length  = args.max_length,
        max_samples = max_samples,
    )

    val_dataset = None
    if args.val_json and args.val_csv and os.path.exists(args.val_json):
        val_dataset = BanglaMedVQADataset(
            json_path   = args.val_json,
            csv_path    = args.val_csv,
            processor   = model.processor,
            max_length  = args.max_length,
            max_samples = max_samples,
        )

    num_workers = 0 if (device == "cpu" or args.debug) else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = (device != "cpu"),
        collate_fn  = collate_fn,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=(device != "cpu"),
                   collate_fn=collate_fn)
        if val_dataset else None
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples: {len(val_dataset)}")

    # ---- Optimizer — separate LRs for VPT, router, LoRA -------------------
    vpt_params    = (list(model.prompt_learner.gen_prompts.parameters()) +
                     list(model.prompt_learner.spec_prompts.parameters()))
    router_params = (list(model.router.mlp.parameters())
                     if model.router is not None else [])
    lora_params   = [p for n, p in model.model.named_parameters()
                     if "lora" in n.lower() and p.requires_grad]

    param_groups = [{"params": vpt_params, "lr": args.lr_vpt}]
    if router_params:
        param_groups.append({"params": router_params, "lr": args.lr_router})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_lora})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Loss curve log file -----------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    loss_log_path = os.path.join(args.output_dir, "train_loss.csv")
    with open(loss_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss", "lm_loss",
                                 "router_loss_component", "cat_loss_component"])

    # ---- Training ----------------------------------------------------------
    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):

        # Two-phase: Phase 1 = VPT warmup, Phase 2 = joint
        if epoch == 0:
            logger.info("Phase 1 — VPT warmup (LoRA frozen)")
            for p in lora_params:
                p.requires_grad_(False)
        elif epoch == 1:
            logger.info("Phase 2 — Joint VPT + LoRA")
            for p in lora_params:
                p.requires_grad_(True)

        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            questions      = batch["question"]
            categories     = batch["category"]
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            oracle_cats = categories if args.use_oracle_router else None

            outputs, router_alpha = model(
                pixel_values      = pixel_values,
                input_ids         = input_ids,
                attention_mask    = attention_mask,
                questions         = questions,
                labels            = labels,
                oracle_categories = oracle_cats,
                image_grid_thw    = image_grid_thw,
            )

            if router_alpha is None or args.use_oracle_router:
                # A1 (no router) or A5 (oracle): no router losses to apply
                loss = outputs.loss
            else:
                loss = compute_loss(
                    lm_loss        = outputs.loss,
                    router_alpha   = router_alpha,
                    category_labels= categories,
                    lambda_router  = args.lambda_router,
                    mu_cat         = args.mu_cat,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1} step {step+1} | "
                    f"loss={loss.item():.4f} lm={outputs.loss.item():.4f}"
                )
                with open(loss_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch + 1, global_step,
                        round(loss.item(), 6),
                        round(outputs.loss.item(), 6),
                        round(args.lambda_router, 6),
                        round(args.mu_cat, 6),
                    ])

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # ---- Validation router accuracy ------------------------------------
        if (val_loader is not None and not args.use_oracle_router
                and model.router is not None):
            model.eval()
            router_acc = evaluate_router(model, val_loader, device)
            logger.info(f"Val router accuracy: {router_acc*100:.2f}%")
            model.train()

        # ---- Save checkpoint -----------------------------------------------
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_payload = {
            "gen_prompts":     model.prompt_learner.gen_prompts.detach().cpu(),
            "spec_prompts":    model.prompt_learner.spec_prompts.detach().cpu(),
            "num_gen_tokens":  model.prompt_learner.num_gen_tokens,
            "num_spec_tokens": model.prompt_learner.num_spec_tokens,
            "epoch":           epoch + 1,
            "args":            vars(args),
        }
        if model.router is not None:
            ckpt_payload["router_mlp"] = model.router.mlp.state_dict()
        torch.save(ckpt_payload, os.path.join(ckpt_dir, "vpt_router.pt"))
        if args.add_lora and hasattr(model.model, "save_pretrained"):
            model.model.save_pretrained(ckpt_dir)
        logger.info(f"Checkpoint saved to {ckpt_dir}")

    logger.info(f"Training complete. Outputs: {args.output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping at the top level.")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical VPT training for Medical VQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument("--config", default=None,
                        help="Optional YAML config (configs/vpt/A*.yaml). "
                             "CLI flags override values in the config.")

    # Model
    parser.add_argument("--model_name",      default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--num_vit_layers",  type=int, default=None,
                        help="Manual override for ViT layer count (auto-detected if None)")

    # Data
    parser.add_argument("--train_json",  default="data/chest_x-ray/train/chest_x-ray_dataset.json")
    parser.add_argument("--train_csv",   default="data/chest_x-ray/train/chest_x-ray.csv")
    parser.add_argument("--val_json",    default="data/chest_x-ray/val/chest_x-ray_dataset.json",
                        help="Optional validation JSON (produced by prepare_val_split.py)")
    parser.add_argument("--val_csv",     default="data/chest_x-ray/val/chest_x-ray.csv")

    # Output
    parser.add_argument("--output_dir",  default="output/vpt_A4")
    parser.add_argument("--log_every",   type=int, default=10)

    # Training
    parser.add_argument("--num_epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--max_length",      type=int,   default=512)
    parser.add_argument("--num_gen_tokens",  type=int,   default=16)
    parser.add_argument("--num_spec_tokens", type=int,   default=24)
    parser.add_argument("--lr_vpt",          type=float, default=2e-4)
    parser.add_argument("--lr_router",       type=float, default=1e-3)
    parser.add_argument("--lr_lora",         type=float, default=1e-4)
    parser.add_argument("--lambda_router",   type=float, default=0.01,
                        help="Router entropy regularisation weight")
    parser.add_argument("--mu_cat",          type=float, default=0.1,
                        help="Category supervision weight")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Trade compute for memory (recommended for 7B on A100 80GB)")

    # LoRA / router flags
    parser.add_argument("--add_lora",          action=argparse.BooleanOptionalAction, default=True,
                        help="Add LoRA to the language model (--add_lora / --no-add_lora)")
    parser.add_argument("--use_oracle_router", action="store_true", default=False,
                        help="Ablation A5: bypass learned router with ground-truth categories")
    parser.add_argument("--disable_router",    action="store_true", default=False,
                        help="Ablation A1: disable router entirely; use 0.5 blend (skips MLP)")

    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device string: 'cuda', 'cuda:0', 'cpu'")

    # ---- Debug mode --------------------------------------------------------
    debug_group = parser.add_argument_group("Debug / CPU smoke-test")
    debug_group.add_argument(
        "--debug", action="store_true", default=False,
        help=(
            "CPU smoke-test: tiny dataset, 1 epoch, float32, LoRA disabled. "
            "Verifies the training loop and gradient flow — NOT for benchmarking. "
            "Use a small model (e.g. Qwen/Qwen2.5-VL-3B-Instruct) for speed."
        ),
    )
    debug_group.add_argument(
        "--debug_samples", type=int, default=10,
        help="Number of samples to use in debug mode",
    )

    args = parser.parse_args()

    # Merge YAML config (CLI flags take precedence)
    if args.config is not None:
        import sys
        yaml_cfg = _load_yaml_config(args.config)
        cli_provided = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                cli_provided.add(tok.lstrip("-").replace("no-", "").split("=")[0])
        for k, v in yaml_cfg.items():
            if k not in cli_provided and hasattr(args, k):
                setattr(args, k, v)

    # Apply debug overrides
    if args.debug:
        args.device         = "cpu"
        args.num_epochs     = 1
        args.batch_size     = 1
        args.num_gen_tokens = 2
        args.num_spec_tokens= 2
        args.add_lora       = False
        args.gradient_checkpointing = False
        args.log_every      = 1
        logger.info(
            f"Debug mode: {args.debug_samples} samples, 1 epoch, CPU, "
            f"no LoRA, 2 gen/spec tokens"
        )

    train(args)


if __name__ == "__main__":
    main()
