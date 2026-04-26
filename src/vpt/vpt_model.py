"""
HierarchicalVPTModel — wraps a VLM with VPT prompt pools and a category router.

Supports:
  - Qwen2.5-VL-7B-Instruct
  - google/medgemma-4b-it
  - Any AutoModelForCausalLM + AutoProcessor VLM

Training components:
  - VPTPromptLearner: learnable Gen/Spec token pools injected into frozen ViT
  - CategoryRouter: MLP that routes questions to pool selection weight alpha
  - LoRA adapters: optional, applied to the language model (identical to baseline)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor

from .prompt_learner import VPTPromptLearner
from .category_router import CategoryRouter


class HierarchicalVPTModel(nn.Module):
    """
    VLM + VPT prompt pools + category router.

    Args:
        model_name:       HuggingFace model ID
        lora_config:      dict with lora_rank/lora_alpha/lora_dropout keys, or None
        num_gen_tokens:   learnable tokens per layer in the Gen pool
        num_spec_tokens:  learnable tokens per layer in the Spec pool
        add_lora:         whether to apply LoRA to the language model
        torch_dtype:      dtype for model weights (bfloat16 on GPU, float32 on CPU)
        device_map:       passed to AutoModelForCausalLM.from_pretrained
        num_vit_layers:   manual override for ViT layer count (auto-detected if None)
    """

    def __init__(
        self,
        model_name: str,
        lora_config: dict = None,
        num_gen_tokens: int = 16,
        num_spec_tokens: int = 24,
        add_lora: bool = True,
        torch_dtype=None,
        device_map: str = "auto",
        num_vit_layers: int = None,
        disable_router: bool = False,
    ):
        super().__init__()
        self.disable_router = disable_router

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if add_lora and lora_config is not None:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_cfg = LoraConfig(
                r              = lora_config.get("lora_rank", 16),
                lora_alpha     = lora_config.get("lora_alpha", 32),
                lora_dropout   = lora_config.get("lora_dropout", 0.05),
                target_modules = "all-linear",
                task_type      = TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, peft_cfg)
        else:
            self.model = base_model

        self._freeze_visual_encoder()

        prompt_dim = self._get_vit_hidden_dim()
        # Use the underlying (un-PEFT-wrapped) model when registering hooks
        # so the hooks fire on the actual ViT modules.
        vit_target = self._unwrap_peft(self.model)
        self.prompt_learner = VPTPromptLearner(
            vit_model       = vit_target,
            num_gen_tokens  = num_gen_tokens,
            num_spec_tokens = num_spec_tokens,
            prompt_dim      = prompt_dim,
            num_layers      = num_vit_layers,
        )
        self.prompt_learner.register_hooks(vit_target)

        # Skip the heavy sentence encoder when the router is unused (A1).
        self.router = None if disable_router else CategoryRouter()

    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_peft(model):
        """Return the underlying base model when wrapped by PEFT, else model."""
        # PeftModel.base_model.model is the original HuggingFace model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            return model.base_model.model
        return model

    def _freeze_visual_encoder(self):
        target = self._unwrap_peft(self.model)
        # Qwen2.5-VL
        if hasattr(target, 'visual'):
            for p in target.visual.parameters():
                p.requires_grad = False
        # MedGemma / SigLIP
        elif hasattr(target, 'vision_tower'):
            for p in target.vision_tower.parameters():
                p.requires_grad = False

    def _get_vit_hidden_dim(self) -> int:
        target = self._unwrap_peft(self.model)
        if hasattr(target, 'visual'):
            return target.visual.config.hidden_size
        return target.vision_tower.config.hidden_size

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        questions: list,
        labels=None,
        oracle_categories: list = None,
        image_grid_thw=None,
    ):
        """
        oracle_categories: ground-truth category strings for ablation A5 (oracle router).
                           If provided, bypasses the learned router.
        image_grid_thw:    Qwen2.5-VL requires this tensor for positional encoding.
        """
        if oracle_categories is not None:
            # Oracle: hard 0/1, kept in float32 (cast to ViT dtype inside hook).
            alpha = torch.tensor(
                [1.0 if CategoryRouter.category_label(c) == 0 else 0.0
                 for c in oracle_categories],
                device=pixel_values.device,
                dtype=torch.float32,
            )
        elif self.disable_router or self.router is None:
            # A1 ablation: no router, equal blend (alpha=None → 0.5 in hook)
            alpha = None
        else:
            alpha = self.router(questions).to(pixel_values.device)

        self.prompt_learner.set_routing_weight(alpha)

        forward_kwargs = dict(
            pixel_values   = pixel_values,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        return self.model(**forward_kwargs), alpha

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list:
        """All parameters that receive gradients during training."""
        params = (
            list(self.prompt_learner.gen_prompts.parameters()) +
            list(self.prompt_learner.spec_prompts.parameters()) +
            list(self.router.mlp.parameters()) +
            [p for p in self.model.parameters() if p.requires_grad]
        )
        return params

    def print_trainable_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
