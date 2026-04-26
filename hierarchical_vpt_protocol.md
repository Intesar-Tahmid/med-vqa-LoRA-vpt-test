# Experiment Protocol: Hierarchical VPT for Category-Aware Medical VQA

**Based on:** `github.com/AhmedRafid023/med-vqa-LoRA`  
**Backbone models:** `Qwen2.5-VL-7B` · `google/medgemma-4b-it`  
**Dataset:** BanglaMedVQA (ChestX-ray8 + MedICaT, 7,000 QA pairs)  
**Target venue:** ACL 2026 / EMNLP 2025  

---

## 1. Overview

Your existing repo fine-tunes VLMs with LoRA via LLaMA Factory. This protocol layers **Hierarchical Visual Prompt Tuning (VPT)** on top of that codebase. The core addition is small: a `CategoryRouter` MLP and two pools of learnable `nn.Parameter` tensors injected into the frozen ViT encoder.

The reason this is worth doing: the paper's own Table 4 shows that after LoRA fine-tuning, Condition accuracy is 18.5% and Position accuracy is 31.25% — the worst categories by far. LoRA only adapts the language model weights. VPT adapts the visual encoder, which is where the grounding failure actually lives.

**What you are adding to the existing codebase:**
- `src/vpt/prompt_learner.py` — the VPT token pools and forward hook injection
- `src/vpt/category_router.py` — the MLP that routes questions to the correct pool
- `src/vpt/vpt_model.py` — wrapper that combines the existing model with VPT
- `train_vpt.py` — new training script (replaces `train_model.py` for this experiment)
- `configs/vpt/` — new config files for each ablation condition

---

## 2. Repository Structure After Changes

```
med-vqa-lora/
├── src/
│   └── vpt/
│       ├── __init__.py
│       ├── prompt_learner.py      # NEW — VPT token pools + ViT hook injection
│       ├── category_router.py     # NEW — MLP router (question → pool selection)
│       └── vpt_model.py           # NEW — wraps existing model with VPT
├── configs/
│   ├── qwen2.5-vl-7b.yaml         # EXISTING — keep untouched (your LoRA baseline)
│   ├── med-gemma-4b.yaml          # EXISTING — keep untouched
│   └── vpt/
│       ├── A1_shared_no_router.yaml
│       ├── A2_shared_with_router.yaml
│       ├── A3_split_with_router.yaml      # proposed method (VPT only)
│       ├── A4_split_router_lora.yaml      # proposed method (full)
│       └── A5_oracle_router.yaml
├── data/                           # EXISTING — no changes needed
├── prepare_data.py                 # EXISTING — no changes needed
├── train_model.py                  # EXISTING — keep as baseline B0
├── train_vpt.py                    # NEW — VPT training script
├── evaluate.py                     # NEW — per-category LAVE evaluation
└── requirements_vpt.txt            # NEW — additional dependencies
```

---

## 3. New Dependencies

Add to your existing `requirements.txt`, or create `requirements_vpt.txt`:

```
transformers>=4.40.0
peft>=0.10.0
torch>=2.2.0
sentence-transformers>=2.7.0   # for question BERT embeddings in the router
evaluate>=0.4.0
bert-score>=0.3.13
openai>=1.0.0                  # for LAVE scoring via GPT-4.1-mini
pillow>=10.0.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.66.0
```

---

## 4. Core Implementation

### 4.1 VPT Prompt Learner (`src/vpt/prompt_learner.py`)

This is the most important new file. It creates two pools of learnable tokens (Gen pool for Modality/Organ, Spec pool for Abnormality/Condition/Position) and injects them into every transformer layer of the frozen ViT via forward pre-hooks.

```python
import torch
import torch.nn as nn


# Which categories belong to which pool
GEN_CATEGORIES  = {"modality", "organ"}
SPEC_CATEGORIES = {"abnormality", "condition", "position"}


class VPTPromptLearner(nn.Module):
    """
    Creates two pools of learnable prompt tokens and injects them
    into the frozen ViT encoder via register_forward_pre_hook().

    Pool selection is controlled by a routing weight alpha (scalar in [0,1])
    produced by the CategoryRouter:
        alpha=1.0  → pure Gen pool
        alpha=0.0  → pure Spec pool
        0 < alpha < 1 → soft blend (used during training)
    """

    def __init__(self, vit_model, num_gen_tokens=16, num_spec_tokens=24,
                 prompt_dim=1024, num_layers=None):
        super().__init__()

        self.num_gen_tokens  = num_gen_tokens
        self.num_spec_tokens = num_spec_tokens
        self.prompt_dim      = prompt_dim

        # Infer number of transformer layers from the ViT
        # For Qwen2.5-VL the ViT blocks are at model.visual.blocks
        # For MedGemma the ViT blocks are at model.vision_tower.vision_model.encoder.layers
        if num_layers is None:
            num_layers = self._count_vit_layers(vit_model)
        self.num_layers = num_layers

        # Gen pool: shape [num_layers, num_gen_tokens, prompt_dim]
        self.gen_prompts = nn.Parameter(
            torch.zeros(num_layers, num_gen_tokens, prompt_dim)
        )
        # Spec pool: shape [num_layers, num_spec_tokens, prompt_dim]
        self.spec_prompts = nn.Parameter(
            torch.zeros(num_layers, num_spec_tokens, prompt_dim)
        )

        # Initialise with small random values (NOT zeros — zeros give zero gradient)
        nn.init.trunc_normal_(self.gen_prompts,  std=0.02)
        nn.init.trunc_normal_(self.spec_prompts, std=0.02)

        # Store hooks so we can remove them later
        self._hooks = []
        self._current_alpha = None   # set per forward pass by router
        self._layer_idx     = 0      # incremented by hooks

    def _count_vit_layers(self, vit_model):
        """Try to auto-detect number of ViT transformer blocks."""
        # Qwen2.5-VL
        if hasattr(vit_model, 'visual') and hasattr(vit_model.visual, 'blocks'):
            return len(vit_model.visual.blocks)
        # MedGemma / SigLIP / CLIP-style
        if hasattr(vit_model, 'vision_tower'):
            encoder = vit_model.vision_tower.vision_model.encoder
            return len(encoder.layers)
        raise ValueError(
            "Cannot auto-detect ViT layers. Pass num_layers explicitly."
        )

    def register_hooks(self, vit_model):
        """
        Register a forward_pre_hook on each ViT transformer block.
        The hook prepends the blended prompt tokens to the hidden state.
        """
        self._remove_hooks()
        self._layer_idx = 0

        # Get the list of transformer blocks
        if hasattr(vit_model, 'visual') and hasattr(vit_model.visual, 'blocks'):
            blocks = vit_model.visual.blocks          # Qwen2.5-VL
        else:
            blocks = vit_model.vision_tower.vision_model.encoder.layers  # MedGemma

        for i, block in enumerate(blocks):
            layer_i = i  # capture by value
            hook = block.register_forward_pre_hook(
                lambda module, args, li=layer_i: self._inject_tokens(args, li)
            )
            self._hooks.append(hook)

    def _inject_tokens(self, args, layer_idx):
        """
        Prepend blended prompt tokens to the hidden state tensor.
        args[0] is the hidden state of shape [B, seq_len, dim].
        Returns the modified args tuple.
        """
        hidden = args[0]   # [B, seq_len, dim]
        B = hidden.shape[0]
        alpha = self._current_alpha  # [B, 1, 1] or scalar

        # Get tokens for this layer and blend
        gen_tok  = self.gen_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)   # [B, G, D]
        spec_tok = self.spec_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)  # [B, S, D]

        if alpha is not None:
            # Soft blend: alpha * gen + (1 - alpha) * spec
            # Pad shorter pool with zeros so dimensions match for blending
            max_tok = max(self.num_gen_tokens, self.num_spec_tokens)
            g = torch.zeros(B, max_tok, self.prompt_dim, device=hidden.device, dtype=hidden.dtype)
            s = torch.zeros(B, max_tok, self.prompt_dim, device=hidden.device, dtype=hidden.dtype)
            g[:, :self.num_gen_tokens,  :] = gen_tok
            s[:, :self.num_spec_tokens, :] = spec_tok
            blended = alpha * g + (1 - alpha) * s   # [B, max_tok, D]
        else:
            # Hard selection fallback: concatenate both (used in A1 shared baseline)
            blended = torch.cat([gen_tok, spec_tok], dim=1)

        # Prepend prompt tokens to the sequence
        hidden_with_prompts = torch.cat([blended, hidden], dim=1)   # [B, P+seq, D]
        return (hidden_with_prompts,) + args[1:]

    def set_routing_weight(self, alpha):
        """
        Call this before each forward pass with the router output.
        alpha: tensor of shape [B] or [B,1] in [0,1], or a scalar float.
        """
        if isinstance(alpha, torch.Tensor):
            self._current_alpha = alpha.view(-1, 1, 1)
        else:
            self._current_alpha = alpha

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self):
        # This module has no direct forward; it operates via hooks.
        pass
```

### 4.2 Category Router (`src/vpt/category_router.py`)

A lightweight 2-layer MLP that reads the question's sentence embedding and outputs `alpha` — the routing weight. `alpha = 1` means full Gen pool, `alpha = 0` means full Spec pool.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


CATEGORY_TO_LABEL = {
    "modality":     0,   # Gen
    "organ":        0,   # Gen
    "abnormality":  1,   # Spec
    "condition":    1,   # Spec
    "position":     1,   # Spec
}


class CategoryRouter(nn.Module):
    """
    2-layer MLP router.
    Input:  question embedding from a frozen multilingual sentence encoder
    Output: alpha in [0,1]  (1 = Gen pool, 0 = Spec pool)

    Uses 'paraphrase-multilingual-mpnet-base-v2' which handles Bangla.
    """

    def __init__(self, embed_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2'
        )
        # Freeze the sentence encoder — we only train the MLP
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # output in [0,1]
        )

    def forward(self, questions: list[str]) -> torch.Tensor:
        """
        questions: list of B question strings (Bangla or English)
        returns:   alpha tensor of shape [B], values in [0,1]
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(
                questions,
                convert_to_tensor=True,
                show_progress_bar=False
            )  # [B, 768]
        alpha = self.mlp(embeddings).squeeze(-1)   # [B]
        return alpha

    @staticmethod
    def category_label(category: str) -> int:
        """Returns 0 (Gen) or 1 (Spec) given a category string."""
        return CATEGORY_TO_LABEL.get(category.lower(), 1)
```

### 4.3 VPT Model Wrapper (`src/vpt/vpt_model.py`)

This wraps the existing model (loaded the same way as in your `train_model.py`) and adds VPT components on top.

```python
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoProcessor

from .prompt_learner import VPTPromptLearner
from .category_router import CategoryRouter


class HierarchicalVPTModel(nn.Module):
    """
    Wraps an existing LLaMA-Factory-compatible VLM and adds:
      - VPTPromptLearner  (two prompt pools injected via hooks)
      - CategoryRouter    (MLP for pool selection)
      - LoRA adapters     (on the language model, identical to your baseline)
    """

    def __init__(self, model_name: str, lora_config: dict = None,
                 num_gen_tokens: int = 16, num_spec_tokens: int = 24,
                 add_lora: bool = True):
        super().__init__()

        # Load base model (same as your existing run_real_inference)
        self.processor = AutoProcessor.from_pretrained(model_name)
        base_model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Apply LoRA if requested (identical config to your YAML baseline)
        if add_lora and lora_config is not None:
            peft_cfg = LoraConfig(
                r                  = lora_config.get("lora_rank", 16),
                lora_alpha         = lora_config.get("lora_alpha", 32),
                lora_dropout       = lora_config.get("lora_dropout", 0.05),
                target_modules     = "all-linear",
                task_type          = TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, peft_cfg)
        else:
            self.model = base_model

        # Freeze the visual encoder (it will be adapted only via VPT hooks)
        self._freeze_visual_encoder()

        # Attach VPT components
        self.prompt_learner = VPTPromptLearner(
            vit_model      = self.model,
            num_gen_tokens = num_gen_tokens,
            num_spec_tokens= num_spec_tokens,
            prompt_dim     = self._get_vit_hidden_dim(),
        )
        self.prompt_learner.register_hooks(self.model)

        self.router = CategoryRouter()

    # ------------------------------------------------------------------
    def _freeze_visual_encoder(self):
        """Freeze all visual encoder parameters."""
        # Qwen2.5-VL
        if hasattr(self.model, 'visual'):
            for p in self.model.visual.parameters():
                p.requires_grad = False
        # MedGemma
        elif hasattr(self.model, 'vision_tower'):
            for p in self.model.vision_tower.parameters():
                p.requires_grad = False

    def _get_vit_hidden_dim(self):
        """Return the hidden dimension of the ViT (needed for prompt init)."""
        if hasattr(self.model, 'visual'):
            return self.model.visual.config.hidden_size          # Qwen2.5-VL
        return self.model.vision_tower.config.hidden_size        # MedGemma

    # ------------------------------------------------------------------
    def forward(self, pixel_values, input_ids, attention_mask,
                questions: list[str], labels=None,
                oracle_categories: list[str] = None):
        """
        oracle_categories: if provided (ablation A5), bypasses the router
                           and uses the ground-truth category label directly.
        """
        # Compute routing weight
        if oracle_categories is not None:
            # Oracle: alpha=1 for Gen categories, alpha=0 for Spec
            alpha = torch.tensor(
                [1.0 if CategoryRouter.category_label(c) == 0 else 0.0
                 for c in oracle_categories],
                device=pixel_values.device
            )
        else:
            alpha = self.router(questions).to(pixel_values.device)

        # Set routing weight — hooks will use this during the ViT forward pass
        self.prompt_learner.set_routing_weight(alpha)

        # Standard VLM forward
        outputs = self.model(
            pixel_values   = pixel_values,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        return outputs

    def trainable_parameters(self):
        """Returns only the parameters that should be trained."""
        params = []
        params += list(self.prompt_learner.gen_prompts.parameters())
        params += list(self.prompt_learner.spec_prompts.parameters())
        params += list(self.router.mlp.parameters())
        # LoRA parameters inside self.model
        params += [p for p in self.model.parameters() if p.requires_grad]
        return params
```

---

## 5. Training Script (`train_vpt.py`)

This replaces `train_model.py` for the VPT experiments. It uses the same data format (the existing `*_dataset.json` files your `prepare_data.py` already produces — no data changes needed).

```python
#!/usr/bin/env python3
"""
VPT training script for Hierarchical VPT experiment.
Uses the same dataset JSONs produced by prepare_data.py.
"""

import os, json, argparse, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.vpt.vpt_model import HierarchicalVPTModel
from src.vpt.category_router import CATEGORY_TO_LABEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BanglaMedVQADataset(Dataset):
    """
    Reads the existing JSON files produced by prepare_data.py.
    Each entry has: conversations[0]['value'] = question,
                    conversations[1]['value'] = answer,
                    images[0] = image_path
    Also reads the category column from the original CSV for router supervision.
    """

    def __init__(self, json_path: str, csv_path: str, processor, max_length=512):
        with open(json_path, encoding='utf-8') as f:
            self.data = json.load(f)

        # Load category labels from the original CSV
        df = pd.read_csv(csv_path)
        # Build a question → category lookup
        self.cat_lookup = dict(zip(df['question_bn'].astype(str), df['category'].astype(str)))

        self.processor  = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry    = self.data[idx]
        img_path = entry['images'][0]
        question = entry['conversations'][0]['value'].replace('<image>\n', '')
        answer   = entry['conversations'][1]['value']
        category = self.cat_lookup.get(question, 'condition')   # default to Spec if unknown

        image = Image.open(img_path).convert('RGB')

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            image, text, return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=self.max_length, add_special_tokens=False
        )

        # Tokenise the answer for the labels
        answer_ids = self.processor.tokenizer(
            answer, return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=64
        ).input_ids

        return {
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels":         answer_ids.squeeze(0),
            "question":       question,
            "category":       category,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(outputs, labels, router_logits, category_labels,
                 lambda_router=0.01, mu_cat=0.1):
    """
    L_total = L_answer + lambda_router * L_entropy + mu_cat * L_category
    """
    # Answer cross-entropy
    L_answer = outputs.loss

    # Router entropy regularisation (prevent always picking one pool)
    router_probs = torch.stack([router_logits, 1 - router_logits], dim=-1)
    L_entropy = -(router_probs * (router_probs + 1e-8).log()).sum(dim=-1).mean()
    L_router  = -lambda_router * L_entropy   # negative because we MAXIMISE entropy

    # Category supervision (oracle signal from training labels)
    cat_targets = torch.tensor(
        [CATEGORY_TO_LABEL.get(c.lower(), 1) for c in category_labels],
        dtype=torch.float32, device=router_logits.device
    )
    L_cat = F.binary_cross_entropy(router_logits, cat_targets)

    return L_answer + L_router + mu_cat * L_cat


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Build model -------------------------------------------------------
    lora_cfg = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05}
    model = HierarchicalVPTModel(
        model_name     = args.model_name,
        lora_config    = lora_cfg if args.add_lora else None,
        num_gen_tokens = args.num_gen_tokens,
        num_spec_tokens= args.num_spec_tokens,
        add_lora       = args.add_lora,
    )

    # ---- Data --------------------------------------------------------------
    train_dataset = BanglaMedVQADataset(
        json_path  = args.train_json,
        csv_path   = args.train_csv,
        processor  = model.processor,
        max_length = args.max_length,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    # ---- Optimizer — two learning rates ------------------------------------
    vpt_params    = list(model.prompt_learner.parameters())
    router_params = list(model.router.mlp.parameters())
    lora_params   = [p for n, p in model.model.named_parameters()
                     if 'lora' in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": vpt_params,    "lr": args.lr_vpt},
        {"params": router_params, "lr": args.lr_router},
        {"params": lora_params,   "lr": args.lr_lora},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Two-phase training ------------------------------------------------
    # Phase 1 (epoch 1): freeze LoRA, train VPT + router only
    # Phase 2 (epoch 2+): unfreeze LoRA, train jointly

    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):

        # Phase switch
        if epoch == 0:
            logger.info("Phase 1 — VPT warmup (LoRA frozen)")
            for p in lora_params:
                p.requires_grad = False
        elif epoch == 1:
            logger.info("Phase 2 — Joint VPT + LoRA")
            for p in lora_params:
                p.requires_grad = True

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):

            pixel_values   = batch["pixel_values"].to("cuda")
            input_ids      = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels         = batch["labels"].to("cuda")
            questions      = batch["question"]
            categories     = batch["category"]

            # Forward
            outputs = model(
                pixel_values   = pixel_values,
                input_ids      = input_ids,
                attention_mask = attention_mask,
                questions      = questions,
                labels         = labels,
                oracle_categories = categories if args.use_oracle_router else None,
            )

            router_alpha = model.router(questions).to("cuda")
            loss = compute_loss(outputs, labels, router_alpha, categories,
                                lambda_router=args.lambda_router,
                                mu_cat=args.mu_cat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

        # Save checkpoint
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "gen_prompts":  model.prompt_learner.gen_prompts,
            "spec_prompts": model.prompt_learner.spec_prompts,
            "router_state": model.router.mlp.state_dict(),
        }, os.path.join(ckpt_dir, "vpt_router.pt"))
        model.model.save_pretrained(ckpt_dir)   # saves LoRA weights

    logger.info(f"Training complete. Outputs at: {args.output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",      default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_json",      default="data/chest_x-ray/train/chest_x-ray_dataset.json")
    parser.add_argument("--train_csv",       default="data/chest_x-ray/train/chest_x-ray.csv")
    parser.add_argument("--output_dir",      default="output/vpt_A4")
    parser.add_argument("--num_epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--max_length",      type=int,   default=512)
    parser.add_argument("--num_gen_tokens",  type=int,   default=16)
    parser.add_argument("--num_spec_tokens", type=int,   default=24)
    parser.add_argument("--lr_vpt",          type=float, default=2e-4)
    parser.add_argument("--lr_router",       type=float, default=1e-3)
    parser.add_argument("--lr_lora",         type=float, default=1e-4)
    parser.add_argument("--lambda_router",   type=float, default=0.01)
    parser.add_argument("--mu_cat",          type=float, default=0.1)
    parser.add_argument("--add_lora",        action="store_true", default=True)
    parser.add_argument("--use_oracle_router", action="store_true", default=False)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
```

---

## 6. Ablation Conditions

Run these five conditions in order. Each one adds exactly one component so the contribution of each is isolated.

| ID | Name | VPT pool | Router | LoRA | Run command |
|----|------|----------|--------|------|-------------|
| B0 | Paper LoRA baseline | — | — | ✓ | `python train_model.py --config configs/qwen2.5-vl-7b.yaml` |
| A1 | Shared VPT, no router | Shared (concat) | ✗ | ✗ | `python train_vpt.py --output_dir output/A1 --no-add_lora` |
| A2 | Shared VPT + router | Shared | ✓ | ✗ | `python train_vpt.py --output_dir output/A2 --no-add_lora` |
| A3 | Split pools + router | Gen + Spec | ✓ | ✗ | `python train_vpt.py --output_dir output/A3 --no-add_lora` |
| **A4** | **Split + router + LoRA (proposed)** | **Gen + Spec** | **✓** | **✓** | `python train_vpt.py --output_dir output/A4 --add_lora` |
| A5 | Oracle router upper bound | Gen + Spec | Oracle | ✓ | `python train_vpt.py --output_dir output/A5 --add_lora --use_oracle_router` |

Run B0 first and verify your reproduced LAVE scores match Table 4 of the paper (within ±1%) before proceeding.

---

## 7. Evaluation Script (`evaluate.py`)

Your existing `run_real_inference()` in `train_model.py` saves a CSV with `predicted_answer_bn`. This script reads that CSV and computes per-category LAVE scores, Accuracy, and BERTScore — matching the exact format of Tables 4 and 5 in the paper.

```python
#!/usr/bin/env python3
"""
Compute per-category LAVE, Accuracy, BERTScore from a predictions CSV.
Matches the evaluation protocol of BanglaMedVQA paper (Tables 4 & 5).
"""

import os, argparse, json
import pandas as pd
from openai import OpenAI
from bert_score import score as bert_score_fn

CATEGORIES = ["modality", "organ", "abnormality", "condition", "position"]
LAVE_BATCH  = 20   # number of pairs per GPT call


def compute_accuracy(df):
    """Exact string match, case-insensitive, stripped."""
    df = df.copy()
    df["match"] = (
        df["predicted_answer_bn"].str.strip().str.lower() ==
        df["llm_answer_bn"].str.strip().str.lower()
    )
    results = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        results[cat] = round(sub["match"].mean() * 100, 2) if len(sub) > 0 else 0.0
    results["overall"] = round(df["match"].mean() * 100, 2)
    return results


def compute_bertscore(df):
    """BERTScore F1 with xlm-roberta-large (multilingual, handles Bangla)."""
    preds = df["predicted_answer_bn"].fillna("").tolist()
    refs  = df["llm_answer_bn"].fillna("").tolist()
    _, _, F1 = bert_score_fn(preds, refs, lang="bn", verbose=False)
    df = df.copy()
    df["bscore"] = F1.numpy()
    results = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        results[cat] = round(sub["bscore"].mean() * 100, 2) if len(sub) > 0 else 0.0
    results["overall"] = round(df["bscore"].mean() * 100, 2)
    return results


def compute_lave(df, openai_client):
    """
    LLM-Assisted VQA Evaluation using GPT-4.1-mini as judge.
    Matches the LAVE protocol from Mañas et al. 2024.
    """
    preds = df["predicted_answer_bn"].fillna("").tolist()
    refs  = df["llm_answer_bn"].fillna("").tolist()

    scores = []
    for i in range(0, len(preds), LAVE_BATCH):
        batch_preds = preds[i:i+LAVE_BATCH]
        batch_refs  = refs[i:i+LAVE_BATCH]

        pairs = "\n".join([
            f"{j+1}. Reference: {r} | Prediction: {p}"
            for j, (r, p) in enumerate(zip(batch_refs, batch_preds))
        ])

        prompt = (
            "You are a strict evaluator for a medical VQA task in Bangla.\n"
            "For each pair below, output ONLY a JSON list of floats in [0,1].\n"
            "Score 1.0 if the prediction is semantically equivalent to the reference, "
            "0.5 if partially correct, 0.0 if wrong or missing.\n"
            "Do NOT output any text outside the JSON list.\n\n"
            f"{pairs}"
        )

        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        try:
            batch_scores = json.loads(response.choices[0].message.content.strip())
        except Exception:
            batch_scores = [0.0] * len(batch_preds)
        scores.extend(batch_scores)

    df = df.copy()
    df["lave"] = scores[:len(df)]

    results = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        results[cat] = round(sub["lave"].mean() * 100, 2) if len(sub) > 0 else 0.0
    results["overall"] = round(df["lave"].mean() * 100, 2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", required=True,
                        help="Path to predictions CSV from run_real_inference()")
    parser.add_argument("--output_json", default="evaluation_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)

    # Validate required columns
    required = ["predicted_answer_bn", "llm_answer_bn", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. "
                         f"Make sure your CSV has a 'category' column.")

    print("Computing Accuracy...")
    acc = compute_accuracy(df)

    print("Computing BERTScore...")
    bsc = compute_bertscore(df)

    print("Computing LAVE (requires OPENAI_API_KEY)...")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    lave = compute_lave(df, client)

    results = {
        "accuracy":   acc,
        "bertscore":  bsc,
        "lave":       lave,
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty print like Table 4
    print(f"\n{'Category':<14} {'Acc':>6} {'BScore':>8} {'LAVE':>7}")
    print("-" * 40)
    for cat in CATEGORIES + ["overall"]:
        print(f"{cat:<14} {acc[cat]:>6.2f} {bsc[cat]:>8.2f} {lave[cat]:>7.2f}")

    print(f"\nFull results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# After running inference which produces a predictions CSV:
export OPENAI_API_KEY=your_key_here
python evaluate.py \
  --predictions_csv output/A4/chest_x-ray_test_predictions.csv \
  --output_json results/A4_results.json
```

> **Important:** Your existing `run_real_inference()` in `train_model.py` does not currently save the `category` column in the output CSV (it saves it empty — see the `'category': ''` line). You need to fix this by reading `category` from the original CSV and including it in `predictions_data`. This is a one-line fix in `train_model.py`.

---

## 8. Hyperparameter Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_gen_tokens` | 16 per layer | Generalized pool |
| `num_spec_tokens` | 24 per layer | Specialized pool — 1.5× Gen |
| `lora_rank` | 16 | Identical to paper |
| `lora_alpha` | 32 | Identical to paper |
| `lora_dropout` | 0.05 | Slight increase from 0 for regularisation |
| `lr_vpt` | 2e-4 (Phase 1) → 1e-4 (Phase 2) | Reduced in Phase 2 for stability |
| `lr_router` | 1e-3 | Higher — MLP converges quickly |
| `lr_lora` | 1e-4 | Same as paper |
| `batch_size` | 4 (grad_accum=4 → effective 16) | A100 80GB |
| `num_epochs` | 3 (1 warmup + 2 joint) | Phase 1 = epoch 1 only |
| `max_seq_len` | 512 | Identical to paper |
| `lambda_router` | 0.01 | Entropy regularisation weight |
| `mu_cat` | 0.1 | Category supervision weight |
| `warmup_ratio` | 0.05 | Cosine scheduler |
| Sentence encoder | `paraphrase-multilingual-mpnet-base-v2` | Handles Bangla |

---

## 9. Data Split

Your existing `prepare_data.py` already splits into train and test. For the VPT experiment you need a validation split too. Add this to `prepare_data.py` or handle it in `train_vpt.py`:

```
Train:  70%  →  4,900 pairs  (existing train JSON)
Val:    10%  →    700 pairs  (sample from train — stratified by category)
Test:   20%  →  1,400 pairs  (existing test JSON — never touch during training)
```

Use stratified sampling so each category has equal representation in the val split. This is critical for monitoring Condition and Position performance during training, which are the low-frequency failure cases.

---

## 10. Expected Results

Target numbers to beat from **Table 4 of the paper** (Qwen2.5-VL-7B, Bangla, LoRA fine-tuning = your baseline B0):

| Category | Paper (B0) | Target (A4) | Delta |
|----------|-----------|-------------|-------|
| Modality | 96.0% | ~97% | +1 pt |
| Organ | 83.8% | ~87% | +3 pt |
| Abnormality | 75.0% | ~83% | +8 pt |
| **Condition** | **18.5%** | **~38%** | **+20 pt** |
| **Position** | **31.3%** | **~50%** | **+18 pt** |
| Overall | 49.3% | ~62% | +13 pt |

The key novelty claim for the paper is the Condition and Position gains. If A4 improves these by even 10 points over B0 while maintaining Modality/Organ performance, that is a publishable result.

---

## 11. Execution Order

Run in this exact order. Do not skip steps.

```bash
# Step 1: Reproduce baseline (verify your setup matches Table 4)
python train_model.py --config configs/qwen2.5-vl-7b.yaml
python train_model.py --config configs/qwen2.5-vl-7b-test.yaml
python evaluate.py --predictions_csv output/chest_x-ray_test_predictions.csv

# Step 2: Ablation A1 — shared VPT, no router, no LoRA
python train_vpt.py --output_dir output/A1 --no-add_lora --num_gen_tokens 20 --num_spec_tokens 20

# Step 3: Ablation A2 — shared VPT + router, no LoRA
python train_vpt.py --output_dir output/A2 --no-add_lora

# Step 4: Ablation A3 — split pools + router, no LoRA (core VPT contribution)
python train_vpt.py --output_dir output/A3 --no-add_lora

# Step 5: Ablation A4 — full proposed method
python train_vpt.py --output_dir output/A4 --add_lora

# Step 6: Ablation A5 — oracle router upper bound
python train_vpt.py --output_dir output/A5 --add_lora --use_oracle_router

# Step 7: Evaluate all conditions
for COND in B0 A1 A2 A3 A4 A5; do
  python evaluate.py \
    --predictions_csv output/${COND}/chest_x-ray_test_predictions.csv \
    --output_json results/${COND}_results.json
done

# Step 8: Repeat A4 with Med-Gemma backbone
python train_vpt.py \
  --model_name google/medgemma-4b-it \
  --output_dir output/A4_medgemma \
  --add_lora
```

---

## 12. What to Record

For each ablation run, save:

- Full evaluation JSON (`results/A*_results.json`) with per-category Acc / BScore / LAVE
- Training loss curve (log to a CSV per epoch)
- Router accuracy on the val set: what fraction of questions does the router correctly classify as Gen vs. Spec? (This goes into the paper's analysis section)
- GPU memory usage and training time per epoch (for the efficiency claim)
- At least 5 qualitative examples per ablation showing where A4 answers correctly but B0 does not, specifically for Condition and Position questions

---

## 13. Key Implementation Pitfalls

**The ViT layer structure is model-specific.** `VPTPromptLearner._count_vit_layers()` tries to auto-detect, but you may need to manually inspect the model's named modules first:

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
for name, module in model.named_modules():
    print(name)
```

Look for repeating block names like `visual.blocks.0`, `visual.blocks.1`, etc. That pattern tells you the path to pass to the hook registration.

**Token injection changes the sequence length.** When you prepend VPT tokens, the ViT output sequence gets longer by `num_prompt_tokens`. This extended output then goes through the projection layer into the LLM. Make sure the projection layer (Qwen2.5-VL calls this the `visual_merger` or similar) can handle variable-length visual sequences — it typically can, but verify with a small test forward pass before full training.

**The `category` column must be in your predictions CSV.** Fix the empty string in `train_model.py` line `'category': ''` by loading the category from the original CSV during inference. Without this, `evaluate.py` cannot compute per-category metrics.

**Gradient checkpointing.** If you run out of memory on A100 80GB with both VPT tokens and LoRA active, enable `model.model.gradient_checkpointing_enable()` before training. This trades compute for memory and is standard practice for 7B models.
