# experiments/kaghf.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder
from experiments.kernel_fusion import RandomFourierFeatures  # reuse your RFF module


# -------------------------
# helper: masked mean pool
# -------------------------
def masked_mean(h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    h: (B, T, D)
    valid: (B, T) bool
    returns: (B, D)
    """
    v = valid.unsqueeze(-1).float()
    summed = (h * v).sum(dim=1)
    denom = v.sum(dim=1).clamp(min=1.0)
    return summed / denom


# -------------------------
# Cross-attention block
# -------------------------
class CrossAttention(nn.Module):
    """
    Multihead attention with optional masks.
    Query attends to Key/Value.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        q: (B, Tq, D)
        k/v: (B, Tk, D)
        key_padding_mask: (B, Tk) True for positions to IGNORE (pytorch convention)
        """
        attn_out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.norm(q + self.drop(attn_out))  # residual
        return out


# -------------------------
# Pair expert: BiGRU over concatenated sequences
# -------------------------
class PairExpertBiGRU(nn.Module):
    """
    Takes two sequences (B, T, D) and returns pooled vector (B, D_out).
    """
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Linear(2 * d_model, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, d_model),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        s1, s2: (B, T, D) aligned length T
        valid: (B, T) bool mask of valid timesteps (True=valid)
        returns: (B, D)
        """
        x = torch.cat([s1, s2], dim=-1)          # (B, T, 2D)
        x = self.in_proj(x)                     # (B, T, D)
        h, _ = self.gru(x)                      # (B, T, 2H)
        pooled = masked_mean(h, valid)          # (B, 2H)
        return self.out_proj(pooled)            # (B, D)


# -------------------------
# Kernel expert (vector-level, reuse your idea)
# -------------------------
class KernelExpert(nn.Module):
    """
    Encoded vectors -> RFF per modality -> weighted concat -> project to D.
    This is basically your KernelFusionSystem, but returning a D vector (not yhat).
    """
    def __init__(
        self,
        d_model: int,
        rff_dim: int = 512,
        gamma_text: float = 1.0,
        gamma_audio: float = 1.0,
        gamma_vision: float = 1.0,
        wt: float = 1/3,
        wa: float = 1/3,
        wv: float = 1/3,
        trainable_rff: bool = False,
        l2norm_before_rff: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.l2norm_before_rff = l2norm_before_rff

        self.rff_text = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_text, trainable=trainable_rff)
        self.rff_audio = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_audio, trainable=trainable_rff)
        self.rff_vision = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_vision, trainable=trainable_rff)

        s = wt + wa + wv
        wt, wa, wv = wt / s, wa / s, wv / s
        self.register_buffer("w_sqrt", torch.tensor([wt**0.5, wa**0.5, wv**0.5], dtype=torch.float32))

        self.proj = nn.Sequential(
            nn.Linear(3 * rff_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
        )

    def forward(self, t: torch.Tensor, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.l2norm_before_rff:
            t = F.normalize(t, dim=-1)
            a = F.normalize(a, dim=-1)
            v = F.normalize(v, dim=-1)

        kt = self.rff_text(t)
        ka = self.rff_audio(a)
        kv = self.rff_vision(v)

        wt, wa, wv = self.w_sqrt[0], self.w_sqrt[1], self.w_sqrt[2]
        fused = torch.cat([wt * kt, wa * ka, wv * kv], dim=1)  # (B, 3*rff_dim)
        return self.proj(fused)                                # (B, d_model)


# -------------------------
# Main system: Kernel-augmented gated hierarchical fusion
# -------------------------
class KAGHFSystem(nn.Module):
    """
    MGHF-style:
      - encode sequences
      - cross-attn: T -> A and T -> V (text-guided)
      - pair experts: TA, TV, AV (BiGRU)
      - kernel expert: K
      - soft gate over experts
      - regression head

    Notes:
      - This uses TEXT as the query (like MGHF), but DOES NOT hard-bias text dominance.
      - The gate learns when to upweight TA (helps neg/weak-neg) and when to upweight TV.
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
        # cross-attn
        n_heads: int = 8,
        # kernel expert
        rff_dim: int = 512,
        gamma_text: float = 1.0,
        gamma_audio: float = 1.0,
        gamma_vision: float = 1.0,
        trainable_rff: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        # encoders
        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in, d_model=d_model, kernel_size=vision_kernel,
            dropout=dropout_enc, use_bottleneck=use_bottleneck_vision
        )

        # cross-attention (text queries, nonverbal keys/values)
        self.attn_ta = CrossAttention(d_model, n_heads=n_heads, dropout=dropout_enc)  # T attends A
        self.attn_tv = CrossAttention(d_model, n_heads=n_heads, dropout=dropout_enc)  # T attends V

        # pair experts (BiGRU)
        self.exp_ta = PairExpertBiGRU(d_model, hidden=d_model, dropout=dropout_head)
        self.exp_tv = PairExpertBiGRU(d_model, hidden=d_model, dropout=dropout_head)
        self.exp_av = PairExpertBiGRU(d_model, hidden=d_model, dropout=dropout_head)

        # kernel expert
        self.exp_k = KernelExpert(
            d_model=d_model,
            rff_dim=rff_dim,
            gamma_text=gamma_text,
            gamma_audio=gamma_audio,
            gamma_vision=gamma_vision,
            trainable_rff=trainable_rff,
            dropout=dropout_head,
        )

        # gate over 4 experts
        # gate input: pooled t,a,v + (optionally) simple cues later
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, 4),
        )

        # prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text_features, audio_features, vision_features, audio_length, vision_length):
        """
        text_features: (B, Tt, text_in)
        audio_features: (B, Ta, audio_in)
        vision_features: (B, Tv, vision_in)
        """
        # ---- sequences from audio/vision, pooled from all ----
        t_vec = self.text_enc(text_features)                 # (B, D) using CLS
        a_seq, a_valid = self.audio_enc(audio_features, audio_length, return_seq=True)   # (B, Ta, D), (B, Ta)
        v_seq, v_valid = self.vision_enc(vision_features, vision_length, return_seq=True)# (B, Tv, D), (B, Tv)

        a_vec = masked_mean(a_seq, a_valid)                 # (B, D)
        v_vec = masked_mean(v_seq, v_valid)                 # (B, D)

        # ---- build a "text sequence" for attention ----
        # TextEncoder currently returns only CLS vector.
        # For cross-attn you want token sequence from upstream (B, Tt, D).
        # If your text_features are already BERT embeddings per token (768),
        # simplest is: project each token with same proj used in TextEncoder.
        # We'll reuse its proj for tokens:
        T_seq = self.text_enc.proj(text_features)           # (B, Tt, D)

        # masks: PyTorch expects True = pad/ignore
        a_kpm = ~a_valid
        v_kpm = ~v_valid

        # ---- cross-attn: get text-conditioned nonverbal aligned to T tokens ----
        T_a = self.attn_ta(q=T_seq, k=a_seq, v=a_seq, key_padding_mask=a_kpm)  # (B, Tt, D)
        T_v = self.attn_tv(q=T_seq, k=v_seq, v=v_seq, key_padding_mask=v_kpm)  # (B, Tt, D)

        # ---- pair experts ----
        # Need a valid mask for T_seq (assume all tokens valid; if you have text lengths, add mask)
        B, Tt, _ = T_seq.shape
        t_valid = torch.ones(B, Tt, device=T_seq.device, dtype=torch.bool)

        e_ta = self.exp_ta(T_seq, T_a, t_valid)   # (B, D)
        e_tv = self.exp_tv(T_seq, T_v, t_valid)   # (B, D)

        # For AV expert we need aligned length. Easiest: pool A_t and V_t then treat as length=1 sequence.
        A_pool = masked_mean(a_seq, a_valid).unsqueeze(1)  # (B, 1, D)
        V_pool = masked_mean(v_seq, v_valid).unsqueeze(1)  # (B, 1, D)
        one_valid = torch.ones(B, 1, device=T_seq.device, dtype=torch.bool)
        e_av = self.exp_av(A_pool, V_pool, one_valid)      # (B, D)

        # ---- kernel expert ----
        e_k = self.exp_k(t_vec, a_vec, v_vec)              # (B, D)

        # ---- gate ----
        gate_in = torch.cat([t_vec, a_vec, v_vec], dim=-1) # (B, 3D)
        w = F.softmax(self.gate(gate_in), dim=-1)          # (B, 4)

        # weighted sum of experts
        H = (
            w[:, 0:1] * e_k +
            w[:, 1:2] * e_ta +
            w[:, 2:3] * e_tv +
            w[:, 3:4] * e_av
        )                                                  # (B, D)

        yhat = self.head(H).squeeze(-1)
        return yhat
