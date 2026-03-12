import torch
import torch.nn as nn

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

