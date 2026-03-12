# experiments/attn_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnFusionModel(nn.Module):
    """
    Attention-based fusion model that keeps the SAME regression head as EarlyFusionModel.

    - Compute modality attention weights over {text, audio, vision}
    - Weighted sum -> fused (B, d_model)
    - Repeat fused 3x to match early-fusion head input dim (3*d_model)
    - Pass through the exact same fusion_layers MLP
    """
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()

        # Attention params (inspired by TF self_attention: tanh(Wx) then dot with u)
        self.attn_proj = nn.Linear(d_model, d_model)
        self.u = nn.Parameter(torch.randn(d_model) * 0.01)

        # SAME head structure as EarlyFusionModel
        total_dim = 3 * d_model
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text_features, audio_features, visual_features, return_alphas: bool = False):
        """
        Inputs: (B, d_model) each
        Returns:
          sentiment: (B, 1)
          optionally alphas: (B, 3) for [text, audio, vision]
        """
        # (B, 3, D)
        m = torch.stack([text_features, audio_features, visual_features], dim=1)

        # scores: (B, 3)
        h = torch.tanh(self.attn_proj(m))                 # (B,3,D)
        scores = torch.einsum("bmd,d->bm", h, self.u)     # (B,3)
        alphas = F.softmax(scores, dim=1)                 # (B,3)
        print('Attention weights:')
        print('Text weight:', alphas[:, 0])
        print('Audio weight:', alphas[:, 1])
        print('Vision weight:', alphas[:, 2])
        # fused: (B, D)
        fused = torch.einsum("bm,bmd->bd", alphas, m)

        # match early-fusion head input dim: (B, 3D)
        head_in = torch.cat([fused, fused, fused], dim=1)

        sentiment = self.fusion_layers(head_in)

        if return_alphas:
            return sentiment, alphas
        return sentiment

