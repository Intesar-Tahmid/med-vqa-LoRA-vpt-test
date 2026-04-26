"""
VPT Prompt Learner — Deep Visual Prompt Tuning for hierarchical pool routing.

Two learnable parameter pools (Gen for Modality/Organ, Spec for Abnormality/Condition/Position)
are injected into every frozen ViT transformer layer via forward pre-hooks.
A post-hook on the final block strips the prompt tokens before the visual merger.

Deep VPT protocol:
  - Pre-hook on each block: remove previous layer's tokens, inject fresh tokens
  - Post-hook on last block: strip tokens so the visual merger sees original patch tokens
  - Consistent token count across all layers: max(num_gen_tokens, num_spec_tokens)
"""

import torch
import torch.nn as nn

GEN_CATEGORIES  = {"modality", "organ"}
SPEC_CATEGORIES = {"abnormality", "condition", "position"}


class VPTPromptLearner(nn.Module):
    """
    Learnable prompt pools injected into frozen ViT layers via hooks.

    alpha routing weight (produced by CategoryRouter):
        alpha = 1.0  → pure Gen pool
        alpha = 0.0  → pure Spec pool
        0 < alpha < 1 → soft blend (during training)
        alpha = None  → equal blend (A1 shared baseline, alpha=0.5)
    """

    def __init__(self, vit_model, num_gen_tokens: int = 16, num_spec_tokens: int = 24,
                 prompt_dim: int = 1024, num_layers: int = None):
        super().__init__()

        self.num_gen_tokens  = num_gen_tokens
        self.num_spec_tokens = num_spec_tokens
        self.prompt_dim      = prompt_dim
        # Always inject this many tokens per layer (consistent seq-len for Deep VPT)
        self.num_prompt_tokens = max(num_gen_tokens, num_spec_tokens)

        if num_layers is None:
            num_layers = self._count_vit_layers(vit_model)
        self.num_layers = num_layers

        # Gen pool: [num_layers, num_gen_tokens, prompt_dim]
        self.gen_prompts = nn.Parameter(
            torch.zeros(num_layers, num_gen_tokens, prompt_dim)
        )
        # Spec pool: [num_layers, num_spec_tokens, prompt_dim]
        self.spec_prompts = nn.Parameter(
            torch.zeros(num_layers, num_spec_tokens, prompt_dim)
        )

        nn.init.trunc_normal_(self.gen_prompts,  std=0.02)
        nn.init.trunc_normal_(self.spec_prompts, std=0.02)

        self._hooks: list = []
        self._current_alpha = None  # set per forward pass

    # ------------------------------------------------------------------
    def _count_vit_layers(self, vit_model) -> int:
        # Qwen2.5-VL
        if hasattr(vit_model, 'visual') and hasattr(vit_model.visual, 'blocks'):
            return len(vit_model.visual.blocks)
        # MedGemma / SigLIP / CLIP-style
        if hasattr(vit_model, 'vision_tower'):
            return len(vit_model.vision_tower.vision_model.encoder.layers)
        raise ValueError(
            "Cannot auto-detect ViT layers. "
            "Inspect model.named_modules() and pass num_layers explicitly."
        )

    def _get_vit_blocks(self, vit_model):
        if hasattr(vit_model, 'visual') and hasattr(vit_model.visual, 'blocks'):
            return list(vit_model.visual.blocks)
        return list(vit_model.vision_tower.vision_model.encoder.layers)

    # ------------------------------------------------------------------
    def register_hooks(self, vit_model):
        """Register Deep VPT hooks on all ViT blocks."""
        self._remove_hooks()
        blocks = self._get_vit_blocks(vit_model)

        for i, block in enumerate(blocks):
            is_last = (i == len(blocks) - 1)

            # Pre-hook: inject (and for i>0, first strip previous layer's tokens)
            h = block.register_forward_pre_hook(
                lambda module, args, li=i: self._inject_tokens(args, li)
            )
            self._hooks.append(h)

            # Post-hook on final block: strip prompt tokens before visual merger
            if is_last:
                h = block.register_forward_hook(
                    lambda module, _args, output: self._strip_tokens(output)
                )
                self._hooks.append(h)

    # ------------------------------------------------------------------
    def _inject_tokens(self, args, layer_idx: int):
        """
        For layer_idx == 0: prepend fresh tokens.
        For layer_idx > 0:  strip previous layer's tokens, then prepend fresh ones.
        This keeps sequence length constant throughout the ViT (Deep VPT).
        """
        hidden = args[0]    # [B, seq_len, dim]
        B = hidden.shape[0]

        # Strip tokens left by the previous layer's injection
        if layer_idx > 0:
            hidden = hidden[:, self.num_prompt_tokens:, :]

        alpha = self._current_alpha

        gen_tok  = self.gen_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)   # [B, G, D]
        spec_tok = self.spec_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)  # [B, S, D]

        # Blend: alpha * gen + (1-alpha) * spec, padded to num_prompt_tokens
        N = self.num_prompt_tokens
        g = torch.zeros(B, N, self.prompt_dim, device=hidden.device, dtype=hidden.dtype)
        s = torch.zeros(B, N, self.prompt_dim, device=hidden.device, dtype=hidden.dtype)
        g[:, :self.num_gen_tokens,  :] = gen_tok
        s[:, :self.num_spec_tokens, :] = spec_tok

        if alpha is None:
            # A1 shared baseline: equal blend (no router)
            blended = 0.5 * g + 0.5 * s
        else:
            # Cast alpha to the ViT hidden dtype (e.g. bfloat16) for the blend.
            # The original alpha tensor (still float32, with grad) remains stored
            # in self._current_alpha and is returned to the loss function.
            a = alpha.to(dtype=hidden.dtype) if isinstance(alpha, torch.Tensor) else alpha
            blended = a * g + (1.0 - a) * s   # [B, N, D]

        hidden_with_prompts = torch.cat([blended, hidden], dim=1)  # [B, N+seq, D]
        return (hidden_with_prompts,) + args[1:]

    def _strip_tokens(self, output):
        """Remove prompt tokens from the last ViT block output before the visual merger."""
        if isinstance(output, tuple):
            hidden = output[0][:, self.num_prompt_tokens:, :]
            return (hidden,) + output[1:]
        return output[:, self.num_prompt_tokens:, :]

    # ------------------------------------------------------------------
    def set_routing_weight(self, alpha):
        """
        Call before each model forward with the router output.
        alpha: [B] or [B,1] tensor in [0,1], or scalar float, or None (A1 shared).
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
        # Operates entirely via hooks — no direct forward needed.
        pass
