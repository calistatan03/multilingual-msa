import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    Expects x: (B, T, D) where token 0 is [CLS].
    Returns z: (B, out_dim) (optional projection).
    """
    def __init__(self, in_dim=768, out_dim=256, use_proj=True, dropout=0.1):
        super().__init__()
        self.use_proj = use_proj
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        ) if use_proj else nn.Identity()

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        # CLS is the FIRST token, not the last
        cls = x[:, 0, :]  # (B, D)
        return self.proj(cls)

