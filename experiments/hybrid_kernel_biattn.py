# experiments/hybrid_kernel_biattn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder


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
        # x: (B, in_dim)
        return self.scale * torch.cos(x @ self.W + self.b)


class TextTokenProjector(nn.Module):
    """
    Projects token-level BERT outputs to d_model for attention.
    Input:  x (B, T, 768)
    Output: z (B, T, d_model)
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttnPool(nn.Module):
    """
    Single-step cross-attention that returns a pooled vector.

    We use one query token (B,1,d) attending over a sequence (B,T,d).
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        query_vec: torch.Tensor,          # (B, d)
        key_value_seq: torch.Tensor,      # (B, T, d)
        key_padding_mask: torch.Tensor,   # (B, T) True=PAD, False=VALID
    ) -> torch.Tensor:
        # shape to (B,1,d)
        q = self.ln_q(query_vec).unsqueeze(1)
        kv = key_value_seq

        out, _ = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.ln_out(out)  # (B,1,d)
        return out.squeeze(1)   # (B,d)


class MLPHead(nn.Module):
    """
    Simple regression head -> (B,)
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


class HybridKernelBiAttnFusionSystem(nn.Module):
    """
    Hybrid fusion for CH-SIMS:
      - Expert 1: Kernel Fusion (RFF) on pooled (t,a,v)
      - Expert 2: Text->Audio cross-attn pooled vector
      - Expert 3: Audio->Text cross-attn pooled vector
      - Gating network mixes expert predictions per sample.

    Inputs:
      text_features:  (B, Tt, 768)   (BERT token embeddings; token 0 is CLS)
      audio_features: (B, Ta, audio_in)
      vision_features:(B, Tv, vision_in)
      audio_length:   (B,)
      vision_length:  (B,)
      (Optional) text_mask: (B,Tt) True=VALID or False=PAD. If None, assumes all valid.
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        dropout_enc: float = 0.3,
        dropout_head: float = 0.3,
        # modality dims
        audio_in: int = 35,
        vision_in: int = 1629,
        text_in: int = 768,
        # encoders
        audio_kernel: int = 3,
        vision_kernel: int = 3,
        use_bottleneck_vision: bool = True,
        # attention
        n_heads: int = 8,
        # kernel fusion params
        rff_dim: int = 512,
        gamma_text: float = 1.0,
        gamma_audio: float = 1.0,
        gamma_vision: float = 1.0,
        trainable_rff: bool = False,
        l2norm_before_rff: bool = True,
        # fusion weights for kernel blocks
        wt: float = 1/3,
        wa: float = 1/3,
        wv: float = 1/3,
        # enable/disable vision in kernel expert
        use_vision_in_kernel: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.l2norm_before_rff = bool(l2norm_before_rff)
        self.use_vision_in_kernel = bool(use_vision_in_kernel)

        # token-level text projector
        self.text_proj = TextTokenProjector(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)

        # audio/vision encoders (sequence + pooled)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        # cross-attn pooling modules
        self.text_to_audio = CrossAttnPool(d_model=d_model, n_heads=n_heads, dropout=dropout_enc)
        self.audio_to_text = CrossAttnPool(d_model=d_model, n_heads=n_heads, dropout=dropout_enc)

        # kernel expert RFFs
        self.rff_text = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_text, trainable=trainable_rff)
        self.rff_audio = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_audio, trainable=trainable_rff)
        self.rff_vision = RandomFourierFeatures(d_model, rff_dim, gamma=gamma_vision, trainable=trainable_rff)

        # normalised sqrt weights for kernel concat blocks
        s = wt + wa + wv
        wt, wa, wv = wt / s, wa / s, wv / s
        self.register_buffer("w_sqrt", torch.tensor([math.sqrt(wt), math.sqrt(wa), math.sqrt(wv)], dtype=torch.float32))

        # --- Expert heads ---
        # 1) Kernel expert head input: 3*rff_dim -> project -> head
        self.kernel_proj = nn.Linear(3 * rff_dim, 3 * d_model)
        self.kernel_head = MLPHead(in_dim=3 * d_model, hidden_dim=hidden_dim, dropout=dropout_head)

        # 2) T->A expert head: concat [t_cls, ta_vec] -> head
        self.ta_head = MLPHead(in_dim=2 * d_model, hidden_dim=hidden_dim, dropout=dropout_head)

        # 3) A->T expert head: concat [a_pool, at_vec] -> head
        self.at_head = MLPHead(in_dim=2 * d_model, hidden_dim=hidden_dim, dropout=dropout_head)

        # --- Gate ---
        # gate uses cheap pooled signals so it can learn:
        #   "use A->T more when audio is informative"
        gate_in = 3 * d_model  # t_cls + a_pool + v_pool
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, 3),   # weights for [kernel, TA, AT]
        )

    @staticmethod
    def _key_padding_from_valid(valid_mask: torch.Tensor) -> torch.Tensor:
        """
        valid_mask: (B,T) True=valid
        returns key_padding_mask: (B,T) True=pad
        """
        return ~valid_mask

    def forward(
        self,
        text_features: torch.Tensor,   # (B, Tt, 768)
        audio_features: torch.Tensor,  # (B, Ta, audio_in)
        vision_features: torch.Tensor, # (B, Tv, vision_in)
        audio_length: torch.Tensor,    # (B,)
        vision_length: torch.Tensor,   # (B,)
        text_valid_mask: torch.Tensor = None,  # optional (B,Tt) True=valid
    ):
        B, Tt, _ = text_features.shape

        # ---- Text token projection ----
        t_seq = self.text_proj(text_features)      # (B, Tt, d)
        t_cls = t_seq[:, 0, :]                     # (B, d)

        if text_valid_mask is None:
            # assume all tokens valid
            text_valid_mask = torch.ones((B, Tt), dtype=torch.bool, device=text_features.device)
        t_kpm = self._key_padding_from_valid(text_valid_mask)  # True=PAD

        # ---- Audio / Vision encoders -> seq + masks + pooled ----
        a_seq, a_valid = self.audio_enc(audio_features, audio_length, return_seq=True)   # (B,Ta,d), (B,Ta)
        v_seq, v_valid = self.vision_enc(vision_features, vision_length, return_seq=True)

        # pooled audio/vision (masked mean)
        a_pool = self.audio_enc(audio_features, audio_length, return_seq=False)          # (B,d)
        v_pool = self.vision_enc(vision_features, vision_length, return_seq=False)       # (B,d)

        a_kpm = self._key_padding_from_valid(a_valid)
        v_kpm = self._key_padding_from_valid(v_valid)

        # ---- Cross-attention pooled vectors ----
        # Text->Audio: CLS attends over audio sequence
        ta_vec = self.text_to_audio(t_cls, a_seq, key_padding_mask=a_kpm)   # (B,d)

        # Audio->Text: pooled audio vector attends over text tokens
        at_vec = self.audio_to_text(a_pool, t_seq, key_padding_mask=t_kpm)  # (B,d)

        # ---- Expert predictions ----
        # (1) Kernel expert uses pooled (t_cls, a_pool, v_pool) in RFF space
        t0, a0, v0 = t_cls, a_pool, v_pool
        if self.l2norm_before_rff:
            t0 = F.normalize(t0, dim=-1)
            a0 = F.normalize(a0, dim=-1)
            v0 = F.normalize(v0, dim=-1)

        kt = self.rff_text(t0)
        ka = self.rff_audio(a0)
        kv = self.rff_vision(v0)

        wt, wa, wv = self.w_sqrt[0], self.w_sqrt[1], self.w_sqrt[2]
        if self.use_vision_in_kernel:
            fused_kernel = torch.cat([wt * kt, wa * ka, wv * kv], dim=1)  # (B,3*rff_dim)
        else:
            # if you want to ablate vision in kernel expert:
            fused_kernel = torch.cat([wt * kt, wa * ka, torch.zeros_like(kv)], dim=1)

        kernel_in = self.kernel_proj(fused_kernel)  # (B,3*d)
        pred_kernel = self.kernel_head(kernel_in)   # (B,)

        # (2) T->A expert
        pred_ta = self.ta_head(torch.cat([t_cls, ta_vec], dim=1))          # (B,)

        # (3) A->T expert
        pred_at = self.at_head(torch.cat([a_pool, at_vec], dim=1))         # (B,)

        # ---- Gating ----
        gate_logits = self.gate(torch.cat([t_cls, a_pool, v_pool], dim=1))  # (B,3)
        gate_w = torch.softmax(gate_logits, dim=-1)                         # (B,3)

        # mixture
        preds = torch.stack([pred_kernel, pred_ta, pred_at], dim=1)         # (B,3)
        yhat = torch.sum(gate_w * preds, dim=1)                             # (B,)

        return yhat
