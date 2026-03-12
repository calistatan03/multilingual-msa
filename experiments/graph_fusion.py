import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder
class GraphFusion(nn.Module):
    """
    Graphical fusion over 3 modalities using TFN-like batching ops (bmm),
    but with message passing instead of tensor (Kronecker) fusion.

    Inputs:
      a, v, t: (B, d)  modality vectors (already pooled + projected)
    Output:
      fused: (B, 3*d)  concatenation of updated nodes [a', v', t']
      attn:  (B, 3, 3) attention weights (sender->receiver), diagonal masked
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
        # TFN "hidden sizes" (keep small to avoid huge fused_dim)
        audio_hidden: int = 16,
        vision_hidden: int = 16,
        text_hidden: int = 16,
        num_layers: int = 2,
        # if True, audio/vision get TFN-like small MLPs (BN+ReLU); text gets a simple projection
        use_prefusion_mlps: bool = True,
        use_bottleneck_vision: bool=True,
    ):

        super().__init__()
        self.d = d_model
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout_enc)

        # Similar to TFN: we first map each node before mixing
        self.node_proj = nn.Linear(d_model, d_model)

        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )


        # Pairwise interaction projection (like "fusion" but pairwise)
        # Takes concat([h_i, h_j]) -> message dim d
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout_enc),
            nn.Linear(2 * d_model, d_model)
        )

        # Attention score per edge: message -> scalar
        self.edge_score = nn.Linear(d_model, 1)

        # Update each node from [self, aggregated_messages]
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout_enc),
            nn.Linear(2 * d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)
        total_dim = d_model * 3 
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_enc),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_enc),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _one_layer(self, H: torch.Tensor):
        """
        H: (B, N=3, d)
        Returns:
          H_new: (B, 3, d)
          attn:  (B, 3, 3) with diag masked
        """
        B, N, d = H.shape
        assert N == 3 and d == self.d

        H0 = self.node_proj(H)  # (B, 3, d)

        # Build all pairwise concat([h_i, h_j]) for i(receiver), j(sender)
        # We want tensor (B, i, j, 2d)
        hi = H0.unsqueeze(2).expand(B, N, N, d)  # (B, 3, 3, d) receiver
        hj = H0.unsqueeze(1).expand(B, N, N, d)  # (B, 3, 3, d) sender
        pair = torch.cat([hi, hj], dim=-1)       # (B, 3, 3, 2d)

        # Pairwise messages m_{i<-j}
        msg = self.pair_mlp(pair)                # (B, 3, 3, d)

        # Mask self edges: i==j
        eye = torch.eye(N, device=H.device, dtype=torch.bool).unsqueeze(0)  # (1,3,3)
        msg = msg.masked_fill(eye.unsqueeze(-1), 0.0)

        # Edge scores -> attention over senders j for each receiver i
        scores = self.edge_score(msg).squeeze(-1)      # (B, 3, 3)
        scores = scores.masked_fill(eye, -1e9)         # mask diag
        attn = F.softmax(scores, dim=-1)               # (B, 3, 3)
        attn = self.drop(attn)

        # Aggregate messages for each receiver i:
        # attn: (B,3,3), msg: (B,3,3,d) -> agg: (B,3,d)
        # We can do bmm by flattening receiver dimension:
        msg_flat = msg.view(B * N, N, d)               # (B*3, 3, d)
        attn_flat = attn.view(B * N, 1, N)             # (B*3, 1, 3)
        agg = torch.bmm(attn_flat, msg_flat).squeeze(1) # (B*3, d)
        agg = agg.view(B, N, d)                        # (B, 3, d)

        # Update node states (residual + norm like TFN post layers)
        upd = self.update(torch.cat([H0, agg], dim=-1)) # (B, 3, d)
        upd = self.drop(upd)
        H_new = self.norm(H0 + upd)

        return H_new, attn

    def forward(self, t: torch.Tensor, a: torch.Tensor, v: torch.Tensor, audio_length: torch.Tensor, vision_length: torch.Tensor):

        # Keep TFN's modality order vibe: audio, video, text
        a = self.audio_enc(a,lengths=audio_length)
        t = self.text_enc(t) 
        v = self.vision_enc(v, lengths=vision_length)
        H = torch.stack([a, v, t], dim=1)  # (B, 3, d)

        attn_last = None
        for _ in range(self.num_layers):
            H, attn_last = self._one_layer(H)

        fused = H.reshape(H.size(0), -1)  # (B, 3*d)

        sentiment = self.fusion_layers(fused)

        return sentiment.squeeze(-1)



