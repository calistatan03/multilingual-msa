import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    """
    Vision encoder for per-frame vision features.

    Input:
      x: (B, T, 1629)
      lengths: (B,)

    Default returns:
      z: (B, d_model)  (masked mean pooled)

    If return_seq=True returns:
      h: (B, T, d_model)
      valid: (B, T) bool mask (True=valid)
    """
    def __init__(
        self,
        in_dim: int = 1629,
        d_model: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_bottleneck: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            conv_in_dim = d_model
        else:
            self.bottleneck = nn.Identity()
            conv_in_dim = in_dim

        self.conv = nn.Conv1d(
            in_channels=conv_in_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, return_seq: bool = False):
        """
        x: (B, T, Din)
        lengths: (B,)
        """
        B, T, _ = x.shape
        lengths = lengths.to(x.device).long()

        # (B, T, Din) -> (B, T, conv_in_dim)
        h = self.bottleneck(x)

        # Conv1D expects (B, C, T)
        h = h.transpose(1, 2)        # (B, C, T)
        h = self.conv(h)             # (B, d_model, T)
        h = h.transpose(1, 2)        # (B, T, d_model)

        h = self.act(h)
        h = self.dropout(h)
        h = self.norm(h)

        # Build valid mask
        lengths_clamped = lengths.clamp(min=0, max=T)
        ar = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = ar < lengths_clamped.unsqueeze(1)  # (B, T) bool

        # Handle zero-length sequences safely
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

        # Masked mean pooling -> (B, d_model)
        valid_f = valid.unsqueeze(-1).float()
        summed = (h * valid_f).sum(dim=1)
        z = summed / lengths_safe.unsqueeze(1).float()
        return z

