import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder
from early_fusion import EarlyFusionModel


class TextTokenProjector(nn.Module):
    """
    Projects token embeddings (B,T,768) -> (B,T,d_model) and builds a validity mask.
    """
    def __init__(self, in_dim=768, d_model=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, d_model),
        )

    def forward(self, x):  # x: (B,T,768)
        H = self.proj(x)   # (B,T,D)
        # valid token mask: treat all-zero token vectors as padding
        valid = (x.abs().sum(dim=-1) > 0)  # (B,T) bool
        return H, valid


class GuidedTemporalAttention(nn.Module):
    """
    Luong 'general' attention:
      score(q, k_t) = q^T W k_t

    q: (B, D)
    H: (B, T, D)
    valid: (B, T) bool

    returns:
      ctx: (B, D)
      w: (B, T)
    """
    def __init__(self, d_model: int, bias: bool = False, drop_in: float = 0.0):
        super().__init__()
        self.key_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop_in = drop_in

    def forward(self, q, H, valid=None):
        H = F.dropout(H, p=self.drop_in, training=self.training)
        K = self.key_proj(H)                     # (B,T,D)
        scores = torch.einsum("bd,btd->bt", q, K) # (B,T)

        if valid is not None:
            scores = scores.masked_fill(~valid, float("-inf"))

        w = F.softmax(scores, dim=-1)            # (B,T)
        ctx = torch.einsum("bt,btd->bd", w, H)   # (B,D)
        return ctx, w


class VisionGuidedAttnSystem(nn.Module):
    """
    Vision-guided attention fusion (mirrors your TextGuidedAttnSystem):

      Query: pooled vision vector z_v
      Attends over:
        - audio frames (H_a) -> z_a_att
        - text tokens (H_t)  -> z_t_att   (uses TextTokenProjector)

      Then feeds (z_t_att, z_a_att, z_v) into the SAME EarlyFusionModel head.
    """
    def __init__(
        self,
        d_model=256,
        hidden_dim=256,
        dropout_enc=0.1,
        dropout_head=0.3,
        audio_kernel=3,
        vision_kernel=3,
        use_bottleneck_vision=True,
    ):
        super().__init__()

        # Encoders
        self.text_enc = TextEncoder(in_dim=768, out_dim=d_model, dropout=dropout_enc)  # still available if you want pooled CLS too
        self.text_tok = TextTokenProjector(in_dim=768, d_model=d_model, dropout=dropout_enc)

        self.audio_enc = AudioEncoder(in_dim=35, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=1629,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        # Vision-guided attentions (same module, different targets)
        self.attn_audio = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)
        self.attn_text = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)

        # SAME regression head as baseline
        self.head = EarlyFusionModel(
            text_dim=d_model, audio_dim=d_model, visual_dim=d_model,
            hidden_dim=hidden_dim, dropout_rate=dropout_head
        )

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        # Vision pooled vector is the query
        z_v = self.vision_enc(vision, vision_lengths)  # (B,D)

        # Attend over audio frames using vision query
        H_a, valid_a = self.audio_enc(audio, audio_lengths, return_seq=True)  # (B,Ta,D), (B,Ta)
        z_a, _ = self.attn_audio(z_v, H_a, valid=valid_a)                    # (B,D)

        # Attend over text tokens using vision query
        H_t, valid_t = self.text_tok(text)                                   # (B,Tt,D), (B,Tt)
        z_t, _ = self.attn_text(z_v, H_t, valid=valid_t)                     # (B,D)

        return self.head(z_t, z_a, z_v).squeeze(-1)
