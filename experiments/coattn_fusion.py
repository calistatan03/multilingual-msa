from early_fusion import EarlyFusionModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperSelfAttention(nn.Module):
    """
    Implements the paper's self_attention(x):
      m = x x^T
      n = softmax(m)
      o = n x
      a = o * x
    x: (B, T, D) -> (B, T, D)
    """
    def forward(self, x):
        # (B,T,T)
        m = torch.matmul(x, x.transpose(1, 2))
        n = F.softmax(m, dim=-1)
        o = torch.matmul(n, x)
        a = o * x
        return a

class BiModalAttention(nn.Module):
    """
    Implements paper's bi_modal_attention(x, y):
      m1 = x y^T ; n1=softmax(m1) ; o1=n1 y ; a1=o1*x
      m2 = y x^T ; n2=softmax(m2) ; o2=n2 x ; a2=o2*y
      return concat(a1, a2) along feature dim
    x: (B, Tx, D), y: (B, Ty, D) -> (B, Tx, 2D) and (B, Ty, 2D) if you want both sides
    Here we return both a1 and a2 (so you can pool each side).
    """
    def forward(self, x, y):
        # x attends to y
        m1 = torch.matmul(x, y.transpose(1, 2))      # (B, Tx, Ty)
        n1 = F.softmax(m1, dim=-1)                   # along Ty
        o1 = torch.matmul(n1, y)                     # (B, Tx, D)
        a1 = o1 * x                                  # (B, Tx, D)

        # y attends to x
        m2 = torch.matmul(y, x.transpose(1, 2))      # (B, Ty, Tx)
        n2 = F.softmax(m2, dim=-1)
        o2 = torch.matmul(n2, x)                     # (B, Ty, D)
        a2 = o2 * y                                  # (B, Ty, D)

        return a1, a2

def masked_mean(x, lengths=None):
    """
    x: (B, T, D)
    lengths: (B,) number of valid frames (optional)
    """
    if lengths is None:
        return x.mean(dim=1)

    B, T, D = x.shape
    device = x.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    mask = (idx < lengths.unsqueeze(1)).float()          # (B, T)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0) # (B, 1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

class CoAttentionFusionSystem(nn.Module):
    """
    Co-attention fusion inspired by the paper.

    Steps:
      1) Project each modality sequence to shared d_model
      2) (Optional) self-attention within each modality
      3) Bi-modal co-attention between (Text, Audio) and (Text, Vision)
      4) Pool to utterance vectors
      5) Concatenate [t_vec, a_vec, v_vec] -> (B, 3*d_model)
      6) SAME regression head as EarlyFusionModel
    """
    def __init__(
        self,
        d_model=256,
        hidden_dim=256,
        dropout=0.3,
        use_self_attn=True,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn

        # Project raw features to shared dim
        self.t_proj = nn.Linear(768, d_model)
        self.a_proj = nn.Linear(25, d_model)
        self.v_proj = nn.Linear(1629, d_model)

        self.self_attn = PaperSelfAttention()
        self.co_attn = BiModalAttention()

        # Keep SAME head architecture/params as your early fusion
        self.head = EarlyFusionModel(
            text_dim=d_model,
            audio_dim=d_model,
            visual_dim=d_model,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        """
        text:   (B, 39, 768)
        audio:  (B, 400, 25)
        vision: (B, 47, 1629)
        """
        # 1) Project to (B, T, d_model)
        T = self.t_proj(text)
        A = self.a_proj(audio)
        V = self.v_proj(vision)

        # 2) Optional intra-modal self-attention (paper-style)
        if self.use_self_attn:
            T = self.self_attn(T)
            A = self.self_attn(A)
            V = self.self_attn(V)

        # 3) Co-attention: Text<->Audio and Text<->Vision
        # (You can also add Audio<->Vision, but start with these two)
        T_ta, A_at = self.co_attn(T, A)   # (B,39,d), (B,400,d)
        T_tv, V_vt = self.co_attn(T, V)   # (B,39,d), (B,47,d)

        # Combine the two text-side outputs (simple average; you can also concat+proj)
        T_fused_seq = 0.5 * (T_ta + T_tv)

        # 4) Pool to utterance vectors
        t_vec = masked_mean(T_fused_seq, lengths=None)           # (B,d)
        a_vec = masked_mean(A_at, lengths=audio_lengths)         # (B,d)
        v_vec = masked_mean(V_vt, lengths=vision_lengths)        # (B,d)

        t_vec = self.dropout(t_vec)
        a_vec = self.dropout(a_vec)
        v_vec = self.dropout(v_vec)

        # 5) SAME early-fusion head (concat then MLP)
        yhat = self.head(t_vec, a_vec, v_vec).squeeze(-1)        # (B,)
        return yhat

