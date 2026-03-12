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
    returns ctx: (B, D), w: (B, T)
    """
    def __init__(self, d_model: int, bias: bool = False, drop_in: float = 0.0):
        super().__init__()
        self.key_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop_in = drop_in

    def forward(self, q, H, valid=None):
        H = F.dropout(H, p=self.drop_in, training=self.training)
        K = self.key_proj(H)                     # (B,T,D)
        scores = torch.einsum("bd,btd->bt", q, K)

        if valid is not None:
            scores = scores.masked_fill(~valid, float("-inf"))

        w = F.softmax(scores, dim=-1)            # (B,T)
        ctx = torch.einsum("bt,btd->bd", w, H)   # (B,D)
        return ctx, w

class AudioGuidedAttnSystem(nn.Module):
    """
    Audio query attends over (text tokens) and (vision frames).
    Output goes into SAME EarlyFusionModel head.
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

        # pooled vectors
        self.audio_enc = AudioEncoder(in_dim=35, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(in_dim=1629, d_model=d_model, kernel_size=vision_kernel,
                                        dropout=dropout_enc, use_bottleneck=use_bottleneck_vision)
        # text pooled vector (optional; useful for head)
        self.text_enc = TextEncoder(in_dim=768, out_dim=d_model, dropout=dropout_enc)
        # text sequence for attention
        self.text_tok = TextTokenProjector(in_dim=768, d_model=d_model, dropout=dropout_enc)

        self.attn_text = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)
        self.attn_vision = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)

        self.head = EarlyFusionModel(
            text_dim=d_model, audio_dim=d_model, visual_dim=d_model,
            hidden_dim=hidden_dim, dropout_rate=dropout_head
        )

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        # Query (audio)
        H_a, valid_a = self.audio_enc(audio, audio_lengths, return_seq=True)   # (B,Ta,D), (B,Ta)
        z_a = (H_a * valid_a.unsqueeze(-1).float()).sum(dim=1) / audio_lengths.clamp(min=1).unsqueeze(1).float()
        # Alternatively: if your AudioEncoder already has a pooled mode, use that instead.

        # Keys/values: text tokens + vision frames
        H_t, valid_t = self.text_tok(text)                                     # (B,Tt,D), (B,Tt)
        H_v, valid_v = self.vision_enc(vision, vision_lengths, return_seq=True)# (B,Tv,D), (B,Tv)

        z_t_att, _ = self.attn_text(z_a, H_t, valid=valid_t)                   # (B,D)
        z_v_att, _ = self.attn_vision(z_a, H_v, valid=valid_v)                 # (B,D)

        # Optional: also include a pooled text vector (helps keep parity with other baselines)
        # z_t = self.text_enc(text)                                              # (B,D)

        # Head expects (text,audio,vision) dims; here we treat:
        # text = pooled text (z_t), audio = query audio (z_a), vision = attended vision (z_v_att)
        # and we can swap in attended text if you prefer.
        return self.head(z_t_att, z_a, z_v_att).squeeze(-1)
