import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        positions = torch.arange(max_len).unsqueeze(1)  # (T, 1)
        dimensions = torch.arange(d_model).unsqueeze(0)  # (1, D)

        frequencies = 1 / (10000 ** (2 * (dimensions // 2) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * frequencies[:, 0::2])
        pe[:, 1::2] = torch.cos(positions * frequencies[:, 1::2])

        pe = pe.unsqueeze(0)  # (1, T, D)

        # register as buffer → moves with model, no gradients
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:, :T]

class SinusoidalPositionalEncodingPrev(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]

import torch
import torch.nn as nn

class AudioEncoderPositionEmbedding(nn.Module):
    """
    x: (B, T, Din)
    lengths: (B,) number of valid timesteps (can include 0)

    Default returns:
      z: (B, Dmodel)  (masked mean pooled)

    If return_seq=True returns:
      h: (B, T, Dmodel)  (per-timestep features)
      valid: (B, T) bool mask (True=valid)
    """
    def __init__(self, in_dim: int, d_model: int = 256, kernel_size: int = 3, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, return_seq: bool = False):
        """
        x: (B, T, Din)
        lengths: (B,)
        """
        B, T, _ = x.shape

        # Conv1D expects (B, Din, T)
        h = x.transpose(1, 2)     # (B, Din, T)
        h = self.conv(h)          # (B, Dmodel, T)
        h = h.transpose(1, 2)     # (B, T, Dmodel)

        # Add positional encoding
        h = self.pos_enc(h)

        h = self.act(h)
        h = self.dropout(h)
        h = self.norm(h)

        lengths = lengths.to(x.device)
        lengths_clamped = lengths.clamp(min=0, max=T)

        ar = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = ar < lengths_clamped.unsqueeze(1)

        zero_len = lengths_clamped == 0
        if zero_len.any():
            valid = valid.clone()
            valid[zero_len, 0] = True
            lengths_safe = lengths_clamped.clone()
            lengths_safe[zero_len] = 1
        else:
            lengths_safe = lengths_clamped

        if return_seq:
            return h, valid

        valid_f = valid.unsqueeze(-1).float()
        summed = (h * valid_f).sum(dim=1)
        z = summed / lengths_safe.unsqueeze(1).float()

        return z

class AudioEncoder(nn.Module):
    """
    x: (B, T, Din)
    lengths: (B,) number of valid timesteps (can include 0)

    Default returns:
      z: (B, Dmodel)  (masked mean pooled)

    If return_seq=True returns:
      h: (B, T, Dmodel)  (per-timestep features)
      valid: (B, T) bool mask (True=valid)
    """
    def __init__(self, in_dim: int, d_model: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, return_seq: bool = False):
        """
        x: (B, T, Din)
        lengths: (B,)
        """
        B, T, _ = x.shape

        # Conv1D expects (B, Din, T)
        h = x.transpose(1, 2)     # (B, Din, T)
        h = self.conv(h)          # (B, Dmodel, T)
        h = h.transpose(1, 2)     # (B, T, Dmodel)

        h = self.act(h)
        h = self.dropout(h)
        h = self.norm(h)

        # Build mask: True for valid positions
        lengths = lengths.to(x.device)
        lengths_clamped = lengths.clamp(min=0, max=T)

        ar = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = ar < lengths_clamped.unsqueeze(1)   # (B, T) bool

        # Avoid division by zero: treat zero-length as length 1 and mask first token valid
        zero_len = lengths_clamped == 0
        if zero_len.any():
            valid = valid.clone()
            valid[zero_len, 0] = True
            lengths_safe = lengths_clamped.clone()
            lengths_safe[zero_len] = 1
        else:
            lengths_safe = lengths_clamped

        if return_seq:
            return h, valid

        # Masked mean pooling -> (B, Dmodel)
        valid_f = valid.unsqueeze(-1).float()
        summed = (h * valid_f).sum(dim=1)
        z = summed / lengths_safe.unsqueeze(1).float()

        return z


class AudioEncoderOld(nn.Module):
    """
    x: (B, T, Din)
    lengths: (B,) number of valid timesteps (can include 0)
    Returns z: (B, Dmodel)
    """
    def __init__(self, in_dim: int, d_model: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, Din)
        lengths: (B,)
        """
        B, T, _ = x.shape

        # Conv1D expects (B, Din, T)
        h = x.transpose(1, 2)               # (B, Din, T)
        h = self.conv(h)                    # (B, Dmodel, T)
        h = h.transpose(1, 2)               # (B, T, Dmodel)
        h = self.act(h)
        h = self.dropout(h)
        h = self.norm(h)

        # Build mask: True for valid positions
        # clamp lengths so we don't create invalid masks
        lengths = lengths.to(x.device)
        lengths_clamped = lengths.clamp(min=0, max=T)
        ar = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = ar < lengths_clamped.unsqueeze(1)   # (B, T) bool

        # Avoid division by zero: treat zero-length as length 1 and mask first token valid
        zero_len = lengths_clamped == 0
        if zero_len.any():
            valid[zero_len, 0] = True
            lengths_safe = lengths_clamped.clone()
            lengths_safe[zero_len] = 1
        else:
            lengths_safe = lengths_clamped

        # Masked mean pooling
        valid_f = valid.unsqueeze(-1).float()       # (B, T, 1)
        summed = (h * valid_f).sum(dim=1)           # (B, Dmodel)
        z = summed / lengths_safe.unsqueeze(1).float()

        return z

