# experiments/low_rank_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder


class LowRankFusionSystem(nn.Module):
    """
    LMF-style (low-rank) multimodal fusion that plugs into your pipeline:
      encode -> (optional bottleneck) -> low-rank fusion -> SAME prediction head -> yhat

    Your head expects input dim = 3 * d_model, so this module produces a fused representation
    of shape (B, 3*d_model) and feeds it into your existing fusion_layers head.

    Inputs to forward:
      text_features  : (B, T_text, 768)   (token seq, CLS at index 0 if your TextEncoder uses CLS)
      audio_features : (B, T_a, 35)
      vision_features: (B, T_v, 1629)
      audio_length   : (B,)
      vision_length  : (B,)

    Output:
      yhat: (B,)
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
        # LMF params
        rank: int = 8,
        audio_hidden: int = 16,
        vision_hidden: int = 16,
        text_hidden: int = 16,
        # if True: small MLP bottlenecks (TFN/LMF style), else simple linear bottlenecks
        use_prefusion_mlps: bool = True,
        l2norm_before_fusion: bool = False,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.rank = int(rank)
        self.l2norm_before_fusion = bool(l2norm_before_fusion)

        # 1) Encoders (your existing modules)
        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        # 2) Prefusion bottlenecks to small hidden dims (LMF factorisation is most useful here)
        self.audio_hidden = int(audio_hidden)
        self.vision_hidden = int(vision_hidden)
        self.text_hidden = int(text_hidden)

        if use_prefusion_mlps:
            self.audio_subnet = nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout_enc),
                nn.Linear(d_model, self.audio_hidden),
                nn.ReLU(),
                nn.Linear(self.audio_hidden, self.audio_hidden),
                nn.ReLU(),
            )
            self.vision_subnet = nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout_enc),
                nn.Linear(d_model, self.vision_hidden),
                nn.ReLU(),
                nn.Linear(self.vision_hidden, self.vision_hidden),
                nn.ReLU(),
            )
            self.text_subnet = nn.Sequential(
                nn.Dropout(dropout_enc),
                nn.Linear(d_model, self.text_hidden),
            )
        else:
            self.audio_subnet = nn.Linear(d_model, self.audio_hidden)
            self.vision_subnet = nn.Linear(d_model, self.vision_hidden)
            self.text_subnet = nn.Linear(d_model, self.text_hidden)

        # 3) Low-rank fusion factors
        # We want fused representation for your shared head: out_dim = 3*d_model
        self.fused_out_dim = 3 * self.d_model

        # Factors: (R, hidden+1, out_dim)
        self.audio_factor = nn.Parameter(torch.empty(self.rank, self.audio_hidden + 1, self.fused_out_dim))
        self.vision_factor = nn.Parameter(torch.empty(self.rank, self.vision_hidden + 1, self.fused_out_dim))
        self.text_factor = nn.Parameter(torch.empty(self.rank, self.text_hidden + 1, self.fused_out_dim))

        # Combine ranks: (1, R) and bias: (1, out_dim)
        self.fusion_weights = nn.Parameter(torch.empty(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, self.fused_out_dim))

        # init
        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.vision_factor)
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        # fusion_bias already zeros

        # 4) Your SAME regression head (expects 3*d_model)
        total_dim = 3 * self.d_model
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _prepend_ones(x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> (B, D+1) with leading bias term 1"""
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        return torch.cat([ones, x], dim=1)

    def forward_features(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_length: torch.Tensor,
        vision_length: torch.Tensor,
    ):
        # 1) Encode each modality -> (B, d_model)
        t = self.text_enc(text_features)
        a = self.audio_enc(audio_features, audio_length)
        v = self.vision_enc(vision_features, vision_length)

        if self.l2norm_before_fusion:
            t = F.normalize(t, dim=-1)
            a = F.normalize(a, dim=-1)
            v = F.normalize(v, dim=-1)

        # 2) Bottleneck
        t_h = self.text_subnet(t)
        a_h = self.audio_subnet(a)
        v_h = self.vision_subnet(v)

        # 3) Append ones
        _t = self._prepend_ones(t_h)
        _a = self._prepend_ones(a_h)
        _v = self._prepend_ones(v_h)

        # 4) Low-rank fusion
        fa = torch.matmul(_a, self.audio_factor)
        fv = torch.matmul(_v, self.vision_factor)
        ft = torch.matmul(_t, self.text_factor)

        f_rank = fa * fv * ft
 
        fused = torch.matmul(
            self.fusion_weights, f_rank.permute(1, 0, 2)
        ).squeeze(0) + self.fusion_bias
   
        if fused.dim() == 3:
            if fused.size(1) == 1:
                fused = fused.squeeze(1)
            else:
                fused = fused.reshape(fused.size(0), -1)
        elif fused.dim() == 1:
            fused = fused.unsqueeze(1)

        return fused

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_length: torch.Tensor,
        vision_length: torch.Tensor,
    ):
        # 1) Encode each modality -> (B, d_model)
        t = self.text_enc(text_features)                          # (B, d_model)
        a = self.audio_enc(audio_features, audio_length)          # (B, d_model)
        v = self.vision_enc(vision_features, vision_length)       # (B, d_model)

        if self.l2norm_before_fusion:
            t = F.normalize(t, dim=-1)
            a = F.normalize(a, dim=-1)
            v = F.normalize(v, dim=-1)

        # 2) Bottleneck -> (B, hidden)
        t_h = self.text_subnet(t)     # (B, text_hidden)
        a_h = self.audio_subnet(a)    # (B, audio_hidden)
        v_h = self.vision_subnet(v)   # (B, vision_hidden)

        # 3) Append 1
        _t = self._prepend_ones(t_h)  # (B, text_hidden+1)
        _a = self._prepend_ones(a_h)  # (B, audio_hidden+1)
        _v = self._prepend_ones(v_h)  # (B, vision_hidden+1)

        # 4) Low-rank fusion
        # matmul broadcasts: (B, D) @ (R, D, O) -> (B, R, O)
        fa = torch.matmul(_a, self.audio_factor)   # (B, R, O)
        fv = torch.matmul(_v, self.vision_factor)  # (B, R, O)
        ft = torch.matmul(_t, self.text_factor)    # (B, R, O)

        # elementwise across modalities -> (B, R, O)
        f_rank = fa * fv * ft

        # weighted sum over rank -> (B, O)
        # (1, R) @ (R, B, O) -> (1, B, O) -> (B, O)
        fused = torch.matmul(self.fusion_weights, f_rank.permute(1, 0, 2)).squeeze(0) + self.fusion_bias
        # fused: (B, 3*d_model)

        if fused.dim() == 3:
            # common case: (B, 1, D) -> (B, D)
            if fused.size(1) == 1:
                fused = fused.squeeze(1)
            else:
                # flatten any extra dims
                fused = fused.reshape(fused.size(0), -1)
        elif fused.dim() == 1:
            fused = fused.unsqueeze(1)  # (B,) -> (B,1) (only if you truly expect this)

        # 5) Same head -> (B,)
        yhat = self.fusion_layers(fused)  # (B, 1)
        return yhat.squeeze(-1)
