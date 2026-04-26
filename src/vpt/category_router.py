"""
Category Router — lightweight MLP that maps question embeddings to pool routing weights.

Input:  question strings (Bangla or English)
Output: alpha in [0,1]  (1 = Gen pool, 0 = Spec pool)

Uses 'paraphrase-multilingual-mpnet-base-v2' (768-d) which handles Bangla.
The sentence encoder is frozen; only the MLP is trained.
"""

import torch
import torch.nn as nn


CATEGORY_TO_LABEL = {
    "modality":     0,   # Gen
    "organ":        0,   # Gen
    "abnormality":  1,   # Spec
    "condition":    1,   # Spec
    "position":     1,   # Spec
}

_ENCODER_MODEL = "paraphrase-multilingual-mpnet-base-v2"


class CategoryRouter(nn.Module):
    """2-layer MLP router. Returns alpha per sample (1=Gen, 0=Spec)."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        # Lazy import so the module can be imported without sentence-transformers
        # when the router is disabled (A1 ablation).
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(_ENCODER_MODEL)
            for p in self.encoder.parameters():
                p.requires_grad = False
        except ImportError:
            self.encoder = None

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, questions: list) -> torch.Tensor:
        """
        questions: list of B question strings (Bangla or English)
        returns:   alpha tensor of shape [B], values in [0,1]
        """
        if self.encoder is None:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers>=2.7.0"
            )

        # Determine target device from MLP parameters
        device = next(self.mlp.parameters()).device

        with torch.no_grad():
            embeddings = self.encoder.encode(
                questions,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(device),
            )  # [B, 768]

        alpha = self.mlp(embeddings.to(device)).squeeze(-1)   # [B]
        return alpha

    @staticmethod
    def category_label(category: str) -> int:
        """Returns 0 (Gen) or 1 (Spec) given a category string."""
        return CATEGORY_TO_LABEL.get(category.lower(), 1)
