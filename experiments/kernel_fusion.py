# experiments/kernel_fusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
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
        return self.scale * torch.cos(x @ self.W + self.b)


class KernelFusionSystem(nn.Module):
    """
    Minibatch-friendly kernel-based fusion using Random Fourier Features (RFF).

    Encode -> RFF per modality -> weighted concat -> project -> SAME 3*d_model head.
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

        # Encoders (same as your other systems)
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

        # Map fused kernel features -> (B, 3*d_model) for your head
        self.proj_to_head = nn.Linear(3 * rff_dim, 3 * self.d_model)

        # SAME head structure (expects 3*d_model)
        total_dim = 3 * self.d_model
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
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
        vision_length: torch.Tensor
    ):
        # 1) Encode -> (B, d_model)
        t = self.text_enc(text_features)
        a = self.audio_enc(audio_features, audio_length)
        v = self.vision_enc(vision_features, vision_length)

        # (optional) stabilise gamma by normalising embeddings
        if self.l2norm_before_rff:
            t = F.normalize(t, dim=-1)
            a = F.normalize(a, dim=-1)
            v = F.normalize(v, dim=-1)

        # 2) Kernel feature maps (RFF) -> (B, rff_dim)
        kt = self.rff_text(t)
        ka = self.rff_audio(a)
        kv = self.rff_vision(v)

        # 3) Fuse (weighted concat) -> (B, 3*rff_dim)
        wt, wa, wv = self.w_sqrt[0], self.w_sqrt[1], self.w_sqrt[2]
        fused = torch.cat([wt * kt, wa * ka, wv * kv], dim=1)

        # 4) SAME head path -> (B,)
        head_in = self.proj_to_head(fused)   # (B, 3*d_model)
        yhat = self.fusion_layers(head_in)   # (B, 1)
        return yhat.squeeze(-1)

