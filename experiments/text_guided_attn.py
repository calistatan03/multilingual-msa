import torch
import torch.nn as nn
from early_fusion import EarlyFusionModel
from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedTemporalAttention(nn.Module):
    """
    Luong 'general' attention:
      score(q, k_t) = q^T W k_t

    q: (B, D)
    H: (B, T, D)
    valid: (B, T) bool True=valid

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
        K = self.key_proj(H)                              # (B,T,D)
        scores = torch.einsum("bd,btd->bt", q, K)          # (B,T)

        if valid is not None:
            scores = scores.masked_fill(~valid, float("-inf"))

        w = F.softmax(scores, dim=-1)                     # (B,T)
        ctx = torch.einsum("bt,btd->bd", w, H)             # (B,D)
        return ctx, w


class TextGuidedAttnSystem(nn.Module):
    """
    Text query attends over audio frames and vision frames to produce
    attended utterance vectors, then uses the SAME EarlyFusionModel head.
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

        self.text_enc = TextEncoder(in_dim=768, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=35, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=1629,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        self.attn_audio = TextGuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)
        self.attn_vision = TextGuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)

        # SAME regression head as baseline
        self.head = EarlyFusionModel(
            text_dim=d_model, audio_dim=d_model, visual_dim=d_model,
            hidden_dim=hidden_dim, dropout_rate=dropout_head
        )

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        z_t = self.text_enc(text)  # (B,D)

        H_a, valid_a = self.audio_enc(audio, audio_lengths, return_seq=True)    # (B,Ta,D), (B,Ta)
        H_v, valid_v = self.vision_enc(vision, vision_lengths, return_seq=True) # (B,Tv,D), (B,Tv)

        z_a, _ = self.attn_audio(z_t, H_a, valid=valid_a)   # (B,D)
        z_v, _ = self.attn_vision(z_t, H_v, valid=valid_v)  # (B,D)

        return self.head(z_t, z_a, z_v).squeeze(-1)

