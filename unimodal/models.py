# models.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import math

def last_valid_timestep(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, D]
    lengths: [B] (>=1)
    returns: [B, D] from x[b, lengths[b]-1]
    """
    B, T, D = x.shape
    idx = (lengths.clamp(min=1) - 1).view(B, 1, 1).expand(B, 1, D)  # [B,1,D]
    return x.gather(dim=1, index=idx).squeeze(1)  # [B,D]


@dataclass
class UnimodalConfig:
    in_dim: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    conv_kernel: int = 3
    use_transformer: bool = True  # True for A/V; optional for text_bert

class TextEncoder(nn.Module):
    def __init__(self, in_dim: int = 768, d_model: int = 256, conv_kernel: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (conv_kernel - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, T, 768]
        x = x.transpose(1, 2)   # [B, 768, T]
        x = self.conv(x)        # [B, d_model, T]
        x = x.transpose(1, 2)   # [B, T, d_model]
        x = self.dropout(x)

        lengths_used = lengths.clamp(min=1, max=x.size(1))
        emb = last_valid_timestep(x, lengths_used)   # [B, d_model]
        return x, emb

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Input / output shape: [B, T, D]
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class AVEncoder(nn.Module):
    """
    Acoustic / Visual encoder:
      [B, T, in_dim]
        -> Conv1D projection to d_model
        -> positional encoding
        -> dropout
        -> Transformer encoder
        -> last valid timestep pooling

    Returns:
      seq_out: [B, T, d_model]
      emb:     [B, d_model]
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        conv_kernel: int = 3,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        padding = (conv_kernel - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=padding,
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
        )

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: [B, T, in_dim]
        lengths: [B] true lengths before padding

        Returns:
          seq_out: [B, T, d_model]
          emb: [B, d_model]
        """
        # Project feature dimension -> d_model
        x = x.transpose(1, 2)   # [B, in_dim, T]
        x = self.conv(x)        # [B, d_model, T]
        x = x.transpose(1, 2)   # [B, T, d_model]

        # Add positional information
        x = self.positional_encoding(x)
        x = self.dropout(x)

        B, T, _ = x.shape
        device = x.device
        lengths_used = lengths.clamp(min=1, max=T)

        # Build padding mask for Transformer
        # True means "this position is padding"
        arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)   # [B, T]
        pad_mask = arange >= lengths_used.unsqueeze(1)                      # [B, T]

        # Guard against all-zero-length cases
        zero_len = lengths <= 0
        if zero_len.any():
            pad_mask[zero_len, 0] = False

        # Transformer encoding
        seq_out = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, T, d_model]

        # Last valid timestep pooling
        emb = last_valid_timestep(seq_out, lengths_used)              # [B, d_model]

        return seq_out, emb

class UnimodalEncoder(nn.Module):
    """
    Paper-faithful-ish:
      - map input dim -> d_model via temporal Conv1D
      - optionally Transformer encoder
      - take last valid timestep as utterance embedding
    """
    def __init__(self, cfg: UnimodalConfig):
        super().__init__()
        self.cfg = cfg

        # Conv1D over time: input [B,T,in_dim] -> [B,T,d_model]
        # PyTorch Conv1d expects [B, C, T]
        padding = (cfg.conv_kernel - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=cfg.in_dim,
            out_channels=cfg.d_model,
            kernel_size=cfg.conv_kernel,
            padding=padding,
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.use_transformer = cfg.use_transformer

        if cfg.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                batch_first=True,  # so input is [B,T,D]
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        else:
            self.transformer = None

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,in_dim]
        lengths: [B]
        returns: [B,d_model]
        """
        # Conv1D
        x = x.transpose(1, 2)               # [B,in_dim,T]
        x = self.conv(x)                    # [B,d_model,T]
        x = x.transpose(1, 2)               # [B,T,d_model]
        x = self.dropout(x)
        B, T, _ = x.shape
        device = x.device

        # Use a clamped version of lengths everywhere (pooling + mask)
        lengths_used = lengths.clamp(min=1, max=T)

        # Build padding mask for Transformer (True = pad)
        if self.transformer is not None:
            # Detect zero-length sequences (no valid frames)
            zero_len = lengths <= 0

            arange = torch.arange(T, device=device).view(1, T).expand(B, T)
            pad_mask = arange >= lengths_used.view(B, 1)

            # IMPORTANT: if originally zero-length, force at least one token unmasked
            # to avoid "all-masked" attention -> NaNs.
            if zero_len.any():
                pad_mask[zero_len, 0] = False

            x = self.transformer(x, src_key_padding_mask=pad_mask)
        # Last valid timestep pooling
        emb = last_valid_timestep(x, lengths_used)  # [B,d_model]
        return emb

class SentimentHead(nn.Module):
    """
    Simple regressor:
      [B, d_model] -> [B]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb).squeeze(-1)


class UnimodalRegressor(nn.Module):
    """
    Generic wrapper:
      encoder -> sentiment head

    Works with encoders that return (seq_out, emb).
    """
    def __init__(self, encoder: nn.Module, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.head = SentimentHead(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        seq_out, emb = self.encoder(x, lengths)
        yhat = self.head(emb)
        return {
            "seq_out": seq_out,   # [B, T, d_model]
            "emb": emb,           # [B, d_model]
            "logits": yhat,       # [B]
        }


def build_unimodal_model(modality: str, in_dim: int, d_model: int = 256) -> UnimodalRegressor:
    """
    Convenience builder.

    modality:
      - 'text'
      - 'audio'
      - 'vision'
    """
    if modality == "text":
        encoder = TextEncoder(
                in_dim=in_dim,
                d_model=d_model,
                conv_kernel=1,
                dropout=0.1,
        )
    elif modality in {"audio", "vision"}:
        encoder = AVEncoder(
                in_dim=in_dim,
                d_model=d_model,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
                conv_kernel=3,
                max_seq_len=5000,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return UnimodalRegressor(encoder=encoder, d_model=d_model)
