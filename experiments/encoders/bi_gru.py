import torch
import torch.nn as nn

class BiGRUSequenceEncoder(nn.Module):
    """
    Projects per-step features to d_model, runs a Bi-GRU, and returns:
      H: (B, T, d_model)
      valid: (B, T) bool
    """
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, d_model),
        )
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        """
        x: (B, T, in_dim)
        valid: (B, T) bool
        """
        lengths = valid.sum(dim=1).cpu()

        x = self.in_proj(x)  # (B,T,D)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bigru(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )  # (B,T,D)

        H = self.out_proj(H)
        return H, valid
