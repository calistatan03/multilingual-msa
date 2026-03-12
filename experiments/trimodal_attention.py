import torch
import torch.nn as nn

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder

def make_kpm(lengths, T, device):
    # key_padding_mask: True means PAD
    ar = torch.arange(T, device=device)[None, :]
    return ar >= lengths[:, None]

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_key_padding_mask=None):
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=kv_key_padding_mask)
        x = self.ln1(q + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

class TriFANLikeFusion(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        # Channel 1: text attends to audio + vision
        self.t_from_a = CrossAttnBlock(d_model, n_heads, dropout)
        self.t_from_v = CrossAttnBlock(d_model, n_heads, dropout)

        # Channel 2: A <-> V redundancy reduction
        self.v_from_a = CrossAttnBlock(d_model, n_heads, dropout)
        self.a_from_v = CrossAttnBlock(d_model, n_heads, dropout)

        self.out_ln = nn.LayerNorm(3*d_model)

    def masked_mean(self, x, lengths):
        B, T, D = x.shape
        ar = torch.arange(T, device=x.device)[None, :]
        valid = (ar < lengths[:, None]).unsqueeze(-1).float()
        summed = (x * valid).sum(1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        return summed / denom

    def forward(self, t_seq, a_seq, v_seq, t_len, a_len, v_len):
        a_kpm = make_kpm(a_len, a_seq.size(1), a_seq.device)
        v_kpm = make_kpm(v_len, v_seq.size(1), v_seq.device)

        # Channel 1
        t_a = self.t_from_a(t_seq, a_seq, kv_key_padding_mask=a_kpm)
        t_v = self.t_from_v(t_seq, v_seq, kv_key_padding_mask=v_kpm)
        c1 = t_seq + t_a + t_v

        # Channel 2
        v_a = self.v_from_a(v_seq, a_seq, kv_key_padding_mask=a_kpm)
        a_v = self.a_from_v(a_seq, v_seq, kv_key_padding_mask=v_kpm)

        c1_u = self.masked_mean(c1, t_len)
        v_a_u = self.masked_mean(v_a, v_len)
        a_v_u = self.masked_mean(a_v, a_len)

        fused = self.out_ln(torch.cat([c1_u, v_a_u, a_v_u], dim=1))  # (B, 3D)
        return fused

class TriFANLikeSystem(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256, dropout_enc=0.1, dropout_head=0.3,
                 audio_in=35, vision_in=1629, text_in=768, audio_kernel=3, vision_kernel=3,
                 use_bottleneck_vision=True, n_heads=4):
        super().__init__()

        # --- encoders ---
        self.audio_enc = AudioEncoder(audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)

        # IMPORTANT: you need VisionEncoder to support return_seq=True like AudioEncoder
        self.vision_enc = VisionEncoder(vision_in, d_model=d_model, kernel_size=vision_kernel,
                                        dropout=dropout_enc, use_bottleneck=use_bottleneck_vision)

        # IMPORTANT: you need TextEncoder to optionally return token seq
        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, use_proj=True, dropout=dropout_enc)

        self.fuser = TriFANLikeFusion(d_model=d_model, n_heads=n_heads, dropout=dropout_enc)

        # --- SAME regression head style as early fusion (expects 3*d_model) ---
        self.head = nn.Sequential(
            nn.Linear(3*d_model, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        """
        text: (B, Tt, 768)
        audio: (B, Ta, Da)
        vision: (B, Tv, Dv)
        audio_lengths: (B,)
        vision_lengths: (B,)
        """

        # infer text lengths from padding (assuming padded tokens are all-zeros)
        with torch.no_grad():
            t_valid = (text.abs().sum(dim=-1) > 0)
            text_lengths = t_valid.long().sum(dim=1)

        # encoders must output sequences for attention fusion
        a_seq, _ = self.audio_enc(audio, audio_lengths, return_seq=True)   # (B, Ta, D)

        # TEMP workaround if your VisionEncoder/TextEncoder don't return sequences yet:
        # treat them as length-1 "sequences" (still tri-modal cross-attn, but not over time)
        v_u = self.vision_enc(vision, vision_lengths)                      # (B, D)
        t_u = self.text_enc(text)                                          # (B, D)
        v_seq = v_u.unsqueeze(1)                                           # (B, 1, D)
        t_seq = t_u.unsqueeze(1)                                           # (B, 1, D)
        v_len = torch.ones_like(vision_lengths)
        t_len = torch.ones_like(text_lengths)

        fused = self.fuser(t_seq, a_seq, v_seq, t_len, audio_lengths, v_len)
        return self.head(fused).squeeze(-1)
