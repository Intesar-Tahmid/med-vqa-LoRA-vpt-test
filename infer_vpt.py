#!/usr/bin/env python3
"""
Hierarchical VPT Inference Script

Loads a trained VPT checkpoint (gen/spec prompts + router MLP + optional LoRA)
and runs greedy generation over the test set, producing the same predictions
CSV format that evaluate.py consumes.

Output CSV columns: image_id, image_path, category, question_bn, llm_answer_bn,
                    predicted_answer_bn, alpha (router weight)

Usage:
    python infer_vpt.py \\
        --checkpoint_dir output/A4/checkpoint-epoch3 \\
        --test_json data/chest_x-ray/test/chest_x-ray_dataset.json \\
        --test_csv  data/chest_x-ray/test/chest_x-ray.csv \\
        --output_csv output/A4/chest_x-ray_test_predictions.csv
"""

import os
import json
import argparse
import logging
from typing import Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.vpt.vpt_model import HierarchicalVPTModel
from src.vpt.category_router import CategoryRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vpt_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    add_lora: bool,
    disable_router: bool,
    device: str,
) -> HierarchicalVPTModel:
    """Reconstruct a HierarchicalVPTModel and load trained weights."""
    ckpt_file = os.path.join(checkpoint_dir, "vpt_router.pt")
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"No vpt_router.pt under {checkpoint_dir}")

    payload = torch.load(ckpt_file, map_location="cpu")
    num_gen_tokens  = int(payload.get("num_gen_tokens",  payload["gen_prompts"].shape[1]))
    num_spec_tokens = int(payload.get("num_spec_tokens", payload["spec_prompts"].shape[1]))

    # Reload base model + (re-)attach LoRA from the saved adapter dir if present
    lora_cfg = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05}
    model = HierarchicalVPTModel(
        model_name      = model_name,
        lora_config     = lora_cfg if add_lora else None,
        num_gen_tokens  = num_gen_tokens,
        num_spec_tokens = num_spec_tokens,
        add_lora        = add_lora,
        disable_router  = disable_router,
        device_map      = "auto" if device != "cpu" else {"": "cpu"},
    )

    # Load VPT prompt pools
    with torch.no_grad():
        model.prompt_learner.gen_prompts.copy_(
            payload["gen_prompts"].to(model.prompt_learner.gen_prompts.dtype)
        )
        model.prompt_learner.spec_prompts.copy_(
            payload["spec_prompts"].to(model.prompt_learner.spec_prompts.dtype)
        )

    # Load router MLP
    if model.router is not None and "router_mlp" in payload:
        model.router.mlp.load_state_dict(payload["router_mlp"])

    # Load LoRA adapter weights from the same checkpoint dir
    if add_lora:
        from peft import PeftModel
        adapter_files = [f for f in os.listdir(checkpoint_dir)
                         if f.startswith("adapter_") or f == "adapter_config.json"]
        if adapter_files:
            # The PEFT-wrapped model already exists; load_adapter merges weights in-place.
            try:
                model.model.load_adapter(checkpoint_dir, adapter_name="default")
            except Exception:
                # Fallback: rebuild PeftModel from saved adapter
                base = HierarchicalVPTModel._unwrap_peft(model.model)
                model.model = PeftModel.from_pretrained(base, checkpoint_dir)

    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: HierarchicalVPTModel,
    test_data: list,
    cat_lookup: dict,
    output_csv: str,
    max_new_tokens: int,
    use_oracle_router: bool,
    device: str,
):
    rows = []
    pbar = tqdm(test_data, desc="Inference")

    for i, sample in enumerate(pbar):
        img_path = sample["images"][0]
        question = sample["conversations"][0]["value"].replace("<image>\n", "")
        gt       = sample["conversations"][1]["value"]
        category = cat_lookup.get(question, "")

        try:
            image = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            text = model.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = model.processor(
                text=[text], images=[image],
                return_tensors="pt", add_special_tokens=False,
            ).to(model.model.device if hasattr(model.model, "device") else device)

            # Compute alpha for this sample (oracle or learned router)
            if use_oracle_router:
                alpha = torch.tensor(
                    [1.0 if CategoryRouter.category_label(category) == 0 else 0.0],
                    device=inputs["pixel_values"].device,
                    dtype=torch.float32,
                )
            elif model.router is None:
                alpha = None
            else:
                alpha = model.router([question]).to(inputs["pixel_values"].device)

            model.prompt_learner.set_routing_weight(alpha)

            gen_kwargs = dict(
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                pad_token_id   = model.processor.tokenizer.eos_token_id,
                use_cache      = True,
            )
            output_ids = model.model.generate(**inputs, **gen_kwargs)
            pred = model.processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            alpha_val = (float(alpha.item()) if isinstance(alpha, torch.Tensor)
                         and alpha.numel() == 1 else None)

        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            pred = f"Error: {e}"
            alpha_val = None

        rows.append({
            "image_id":            f"test_{i+1:03d}",
            "image_path":          img_path,
            "category":            category,
            "question_bn":         question,
            "llm_answer_bn":       gt,
            "predicted_answer_bn": pred,
            "alpha":               alpha_val,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical VPT inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to a checkpoint-epoch* directory from train_vpt.py")
    parser.add_argument("--model_name",     default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--test_json",      required=True)
    parser.add_argument("--test_csv",       required=True,
                        help="Original CSV (for question→category lookup)")
    parser.add_argument("--output_csv",     required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--add_lora",       action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_router", action="store_true", default=False)
    parser.add_argument("--use_oracle_router", action="store_true", default=False)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)

    df_csv = pd.read_csv(args.test_csv)
    cat_lookup = dict(zip(df_csv["question_bn"].astype(str),
                          df_csv["category"].astype(str)))

    logger.info(f"Loading checkpoint from {args.checkpoint_dir}")
    model = load_vpt_checkpoint(
        checkpoint_dir = args.checkpoint_dir,
        model_name     = args.model_name,
        add_lora       = args.add_lora,
        disable_router = args.disable_router,
        device         = args.device,
    )

    run_inference(
        model             = model,
        test_data         = test_data,
        cat_lookup        = cat_lookup,
        output_csv        = args.output_csv,
        max_new_tokens    = args.max_new_tokens,
        use_oracle_router = args.use_oracle_router,
        device            = args.device,
    )


if __name__ == "__main__":
    main()
