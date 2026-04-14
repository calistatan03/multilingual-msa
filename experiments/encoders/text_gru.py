from .bi_gru import BiGRUSequenceEncoder
import torch
import torch.nn as nn

class TextBiGRUEncoder(nn.Module):
    def __init__(self, in_dim=768, d_model=256, dropout=0.1):
        super().__init__()
        self.encoder = BiGRUSequenceEncoder(in_dim=in_dim, d_model=d_model, dropout=dropout)

    def forward(self, x):
        valid = (x.abs().sum(dim=-1) > 0)   # (B,T)
        H, valid = self.encoder(x, valid)
        return H, valid
