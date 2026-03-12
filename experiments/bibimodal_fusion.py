# experiments/bi_bimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder


class BiBimodalFusionSystem(nn.Module):
    """
    Bi-bimodal (text-anchored) fusion that plugs into the SAME pipeline style as your TFN system.

    Inputs to forward (raw sequences):
      text_features  : (B, Tt, text_in)   or whatever your TextEncoder expects
      audio_features : (B, Ta, audio_in)
      vision_features: (B, Tv, vision_in)
      audio_length   : (B,)
      vision_length  : (B,)

    Assumptions (matching what you said you already do):
      - TextEncoder returns a pooled CLS vector (B, d_model)
      - AudioEncoder/VisionEncoder return masked-mean pooled vectors (B, d_model)

    Fusion logic (unique to bi-bimodal fusion, close in spirit to your original code):
      - Anchor = text
      - Build two pairwise interactions: (L,A) and (L,V)
      - Compute BOTH directions per pair:
          A->L, L->A, V->L, L->V
      - Concatenate to get fused vector z: (B, 4*d_model)
      - (Optional) run a head (kept similar to your other systems). You can swap this out.

    Notes:
      - This is "fusion-only" at the vector level (pooled vectors),
        but preserves the *bi-bimodal, bidirectional* structure of your original implementation.
      - If you want to exactly reuse your original GatedTransformer-based interactions,
        this module can be adapted to take seq_* as well. For now it stays aligned with your pooled setup.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        dropout_enc: float = 0.3,
        dropout_head: float = 0.3,
        audio_in: int = 35,
        vision_in: int = 1629,
        text_in: int = 768,
        audio_kernel: int = 3,
        vision_kernel: int = 3,
        use_bottleneck_vision: bool = True,
        return_fused: bool = False,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.return_fused = bool(return_fused)

        # -------------------
        # 1) Encoders (same style as your TFN system)
        # -------------------
        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        self.dropout_enc = float(dropout_enc)
        self.dropout_head = float(dropout_head)

        # -------------------
        # 2) Bi-bimodal bidirectional fusion (vector-level)
        # -------------------
        # Each direction uses a small gating/projection block:
        #   x->y : produces a y-shaped vector influenced by x (so output is in y-space, dim=d_model)
        def make_dir_block():
            return nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout_enc),
                nn.Linear(d_model, d_model),
            )

        # text–audio directions
        self.a2l = make_dir_block()  # (a,t) -> l-like vector
        self.l2a = make_dir_block()  # (t,a) -> a-like vector

        # text–vision directions
        self.v2l = make_dir_block()  # (v,t) -> l-like vector
        self.l2v = make_dir_block()  # (t,v) -> v-like vector

        # fused representation: [A->L, L->A, V->L, L->V]
        self.fused_dim = 4 * d_model

        # -------------------
        # 3) Head (kept similar to your other systems; replace with your own if you want)
        # -------------------
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_head),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_head),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_length: torch.Tensor,
        vision_length: torch.Tensor,
        debug: bool = False,
    ):
        # 1) Encode -> pooled vectors (you said you already do CLS + masked mean pooling)
        l = self.text_enc(text_features)                     # (B, d_model)  CLS-like pooled
        a = self.audio_enc(audio_features, audio_length)     # (B, d_model)  masked-mean pooled
        v = self.vision_enc(vision_features, vision_length)  # (B, d_model)  masked-mean pooled

        if debug:
            print("Pooled L:", l.shape)
            print("Pooled A:", a.shape)
            print("Pooled V:", v.shape)

        # 2) Bi-bimodal bidirectional fusion (closest structure to original concat of 4 directions)
        # A->L and L->A
        a2l = self.a2l(torch.cat([a, l], dim=1))  # (B, d_model)
        l2a = self.l2a(torch.cat([l, a], dim=1))  # (B, d_model)

        # V->L and L->V
        v2l = self.v2l(torch.cat([v, l], dim=1))  # (B, d_model)
        l2v = self.l2v(torch.cat([l, v], dim=1))  # (B, d_model)

        fused = torch.cat([a2l, l2a, v2l, l2v], dim=1)  # (B, 4*d_model)

        if self.return_fused:
            # Return fused representation so you can feed it into your own head
            return fused

        # 3) Head (optional)
        yhat = self.fusion_layers(fused)  # (B, 1)
        return yhat.squeeze(-1)

