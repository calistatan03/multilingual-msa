#!/usr/bin/env python3
"""
Mandarin Hybrid Fusion V2 (Kernel Fusion + Bidirectional Audio<->Text Cross-Attention)

What you get:
- Kernel branch: your Random Fourier Feature kernel fusion (text/audio/vision) -> (B, 3*d_model)
- BiCross branch: ATTENTION-BASED pooling using your GuidedTemporalAttention:
    * audio -> text (audio query attends over text tokens)  => z_t_att
    * text  -> audio (text query attends over audio frames) => z_a_att
    * (optional) audio -> vision (audio query attends over vision frames) => z_v_att
    * plus pooled anchors (audio pooled, text pooled)
  then projected to (B, 3*d_model) so it can be gated + mixed with kernel branch
- Single shared prediction head -> yhat (tensor), NOT a tuple

Assumes you already have these in your repo:
  from encoders.text import TextEncoder
  from encoders.audio import AudioEncoder
  from encoders.vision import VisionEncoder

Drop this file somewhere like: experiments/mandarin_hybrid_fusion_v2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder


# -------------------------
# small utils
# -------------------------
def masked_mean_pool(H: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    H: (B,T,D), valid: (B,T) bool
    return: (B,D)
    """
    valid_f = valid.unsqueeze(-1).float()
    denom = valid_f.sum(dim=1).clamp(min=1.0)
    return (H * valid_f).sum(dim=1) / denom


# =========================
# 1) Kernel Fusion (RFF)
# =========================
class RandomFourierFeatures(nn.Module):
    """
    RBF kernel feature map approximation:
      k(x,y)=exp(-gamma||x-y||^2) ≈ phi(x)^T phi(y)
    phi(x) = sqrt(2/M) * cos(xW + b)
    W ~ N(0, 2*gamma I), b ~ Uniform(0, 2pi)
    """
    def __init__(self, in_dim: int, out_dim: int, gamma: float = 1.0, trainable: bool = False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.gamma = float(gamma)

        W = torch.randn(self.in_dim, self.out_dim) * math.sqrt(2.0 * self.gamma)
        b = torch.rand(self.out_dim) * (2.0 * math.pi)

        if trainable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)

        self.scale = math.sqrt(2.0 / self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(x @ self.W + self.b)


class KernelFusionSystemV2(nn.Module):
    """
    Same idea as your KernelFusionSystem, but exposes forward_features() returning (B, 3*d_model),
    and forward() returns yhat only (B,).
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
        # RFF params
        rff_dim: int = 512,
        gamma_text: float = 1.0,
        gamma_audio: float = 1.0,
        gamma_vision: float = 1.0,
        # fusion weights (normalised internally)
        wt: float = 1/3,
        wa: float = 1/3,
        wv: float = 1/3,
        trainable_rff: bool = False,
        l2norm_before_rff: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.dropout_head = float(dropout_head)
        self.l2norm_before_rff = bool(l2norm_before_rff)

        # Encoders
        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        # RFF per modality (d_model -> rff_dim)
        self.rff_text = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_text, trainable=trainable_rff)
        self.rff_audio = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_audio, trainable=trainable_rff)
        self.rff_vision = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_vision, trainable=trainable_rff)

        # Normalise weights; apply sqrt weights to feature blocks
        s = wt + wa + wv
        wt, wa, wv = wt / s, wa / s, wv / s
        self.register_buffer("w_sqrt", torch.tensor([math.sqrt(wt), math.sqrt(wa), math.sqrt(wv)], dtype=torch.float32))

        # Map fused kernel features -> (B, 3*d_model)
        self.proj_to_head = nn.Linear(3 * rff_dim, 3 * self.d_model)

        # Prediction head (kept same structure)
        self.head = SharedPredictionHead(in_dim=3 * self.d_model, hidden_dim=hidden_dim, dropout=dropout_head)

    def forward_features(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_length: torch.Tensor,
        vision_length: torch.Tensor
    ) -> torch.Tensor:
        # Encode -> (B, d_model)
        t = self.text_enc(text_features)
        a = self.audio_enc(audio_features, audio_length)  # pooled
        v = self.vision_enc(vision_features, vision_length)  # pooled

        if self.l2norm_before_rff:
            t = F.normalize(t, dim=-1)
            a = F.normalize(a, dim=-1)
            v = F.normalize(v, dim=-1)

        # RFF -> (B, rff_dim)
        kt = self.rff_text(t)
        ka = self.rff_audio(a)
        kv = self.rff_vision(v)

        # weighted concat -> (B, 3*rff_dim)
        wt, wa, wv = self.w_sqrt[0], self.w_sqrt[1], self.w_sqrt[2]
        fused = torch.cat([wt * kt, wa * ka, wv * kv], dim=1)

        # project -> (B, 3*d_model)
        return self.proj_to_head(fused)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_length: torch.Tensor,
        vision_length: torch.Tensor
    ) -> torch.Tensor:
        feats = self.forward_features(text_features, audio_features, vision_features, audio_length, vision_length)
        yhat = self.head(feats)  # (B,)
        return yhat


# ==========================================
# 2) Bidirectional Audio<->Text Cross-Attn
# ==========================================
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
        self.drop_in = float(drop_in)

    def forward(self, q, H, valid=None):
        H = F.dropout(H, p=self.drop_in, training=self.training)
        K = self.key_proj(H)                     # (B,T,D)
        scores = torch.einsum("bd,btd->bt", q, K)

        if valid is not None:
            scores = scores.masked_fill(~valid, float("-inf"))

        w = F.softmax(scores, dim=-1)            # (B,T)
        ctx = torch.einsum("bt,btd->bd", w, H)   # (B,D)
        return ctx, w

class BiCrossAttnPooling(nn.Module):
    """
    One-way cross-attention pooling:
      - audio -> text  (A2T): audio query attends over text tokens  => z_t_att
      - text  -> audio (T2A): text query attends over audio frames  => z_a_att
      - audio -> vision (A2V): audio query attends over vision frames => z_v_att_a
      - text  -> vision (T2V): text query attends over vision frames  => z_v_att_t

    Returns concatenated feature vector:
      [z_t_att, z_a_att, z_a_pool, (z_v_att_a), (z_v_att_t), (z_t_pool)]
    """
    def __init__(
        self,
        d_model=256,
        dropout_enc=0.1,
        text_in=768,
        audio_in=35,
        vision_in=1629,
        audio_kernel=3,
        vision_kernel=3,
        use_bottleneck_vision=True,
        include_audio_to_vision=True,
        include_text_to_vision=True,   # NEW
        include_pooled_text=True,
    ):
        super().__init__()
        self.include_audio_to_vision = bool(include_audio_to_vision)
        self.include_text_to_vision = bool(include_text_to_vision)  # NEW
        self.include_pooled_text = bool(include_pooled_text)

        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in, d_model=d_model, kernel_size=vision_kernel,
            dropout=dropout_enc, use_bottleneck=use_bottleneck_vision
        )
        self.text_tok = TextTokenProjector(in_dim=text_in, d_model=d_model, dropout=dropout_enc)
        self.text_pool = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc) if self.include_pooled_text else None

        # audio<->text
        self.attn_a2t = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)
        self.attn_t2a = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)

        # one-way to vision
        self.attn_a2v = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)
        self.attn_t2v = GuidedTemporalAttention(d_model=d_model, drop_in=dropout_enc)  # NEW

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        # ---- audio seq ----
        H_a, valid_a = self.audio_enc(audio, audio_lengths, return_seq=True)     # (B,Ta,D), (B,Ta)
        z_a_pool = masked_mean_pool(H_a, valid_a)                                # (B,D)

        # ---- text seq (+ pooled query if enabled) ----
        H_t, valid_t = self.text_tok(text)                                       # (B,Tt,D), (B,Tt)
        z_t_pool = self.text_pool(text) if self.text_pool is not None else None  # (B,D) or None
        z_t_query = z_t_pool if z_t_pool is not None else masked_mean_pool(H_t, valid_t)  # (B,D)

        # ---- A2T ----
        z_t_att, _ = self.attn_a2t(z_a_pool, H_t, valid=valid_t)                 # (B,D)

        # ---- T2A ----
        z_a_att, _ = self.attn_t2a(z_t_query, H_a, valid=valid_a)                # (B,D)

        feats = [z_t_att, z_a_att, z_a_pool]

        # ---- vision seq (compute once if either A2V or T2V needed) ----
        if self.include_audio_to_vision or self.include_text_to_vision:
            H_v, valid_v = self.vision_enc(vision, vision_lengths, return_seq=True)  # (B,Tv,D), (B,Tv)

            if self.include_audio_to_vision:
                z_v_att_a, _ = self.attn_a2v(z_a_pool, H_v, valid=valid_v)           # (B,D)
                feats.append(z_v_att_a)

            if self.include_text_to_vision:
                z_v_att_t, _ = self.attn_t2v(z_t_query, H_v, valid=valid_v)          # (B,D)
                feats.append(z_v_att_t)

        # ---- optional pooled text ----
        if z_t_pool is not None:
            feats.append(z_t_pool)

        return torch.cat(feats, dim=-1)

# =========================
# 3) Shared head + gating
# =========================
class SharedPredictionHead(nn.Module):
    """
    A single shared regressor head: (B, in_dim) -> (B,)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MandarinHybridFusionV2(nn.Module):
    """
    Final model:
      kernel_features (B,3D) + bicross_features (B,kD -> proj -> 3D)
      gate alpha = sigmoid(g([kernel; bicross]))
      mix = (1-alpha)*kernel + alpha*bicross
      yhat = shared_head(mix)

    forward() returns ONLY yhat (tensor) to avoid your detach() tuple crash.
    """
    def __init__(
        self,
        d_model=256,
        hidden_dim=256,
        dropout_enc=0.1,
        dropout_head=0.3,
        # enc dims
        text_in=768,
        audio_in=35,
        vision_in=1629,
        # conv kernels
        audio_kernel=3,
        vision_kernel=3,
        use_bottleneck_vision=True,
        # kernel fusion params
        rff_dim=512,
        gamma_text=1.0,
        gamma_audio=1.0,
        gamma_vision=1.0,
        wt=1/3, wa=1/3, wv=1/3,
        trainable_rff=False,
        l2norm_before_rff=True,
        # bicross switches
        include_audio_to_vision=True,
        include_text_to_vision=True,
        include_pooled_text=True,
    ):
        super().__init__()

        # kernel branch -> (B,3D)
        self.kernel = KernelFusionSystemV2(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout_enc=dropout_enc,
            dropout_head=dropout_head,
            audio_in=audio_in,
            vision_in=vision_in,
            text_in=text_in,
            audio_kernel=audio_kernel,
            vision_kernel=vision_kernel,
            use_bottleneck_vision=use_bottleneck_vision,
            rff_dim=rff_dim,
            gamma_text=gamma_text,
            gamma_audio=gamma_audio,
            gamma_vision=gamma_vision,
            wt=wt, wa=wa, wv=wv,
            trainable_rff=trainable_rff,
            l2norm_before_rff=l2norm_before_rff,
        )

        # bicross branch -> concat blocks
        self.bicross = BiCrossAttnPooling(
            d_model=d_model,
            dropout_enc=dropout_enc,
            text_in=text_in,
            audio_in=audio_in,
            vision_in=vision_in,
            audio_kernel=audio_kernel,
            vision_kernel=vision_kernel,
            use_bottleneck_vision=use_bottleneck_vision,
            include_audio_to_vision=include_audio_to_vision,
            include_text_to_vision=include_text_to_vision,
            include_pooled_text=include_pooled_text,
        )

        # determine bicross output dim
        # base: z_t_att, z_a_att, z_a_pool = 3 blocks
        # + z_v_att if include_audio_to_vision
        # + z_t_pool if include_pooled_text
        n_blocks = 3 + (1 if include_audio_to_vision else 0) + (1 if include_pooled_text else 0) + (1 if include_text_to_vision else 0)
        bicross_dim = n_blocks * d_model

        # project bicross -> (B,3D) for mixing
        self.proj_bicross_to_3d = nn.Linear(bicross_dim, 3 * d_model)

        # gate over concatenated (kernel 3D + bicross 3D) -> alpha in (0,1)
        self.gate = nn.Sequential(
            nn.Linear(6 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, 1),
        )

        # ONE shared prediction head
        self.head = SharedPredictionHead(in_dim=3 * d_model, hidden_dim=hidden_dim, dropout=dropout_head)

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        f_kernel = self.kernel.forward_features(text, audio, vision, audio_lengths, vision_lengths)   # (B,3D)
        f_bi = self.bicross(text, audio, vision, audio_lengths, vision_lengths)                       # (B,kD)
        f_bi = self.proj_bicross_to_3d(f_bi)                                                          # (B,3D)

        alpha = torch.sigmoid(self.gate(torch.cat([f_kernel, f_bi], dim=-1)))                         # (B,1)
        f_mix = (1.0 - alpha) * f_kernel + alpha * f_bi                                               # (B,3D)

        yhat = self.head(f_mix)                                                                       # (B,)
        return yhat

