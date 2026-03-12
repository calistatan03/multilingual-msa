# experiments/tensor_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder

class TensorFusionSystem(nn.Module):
    """
    TFN-style tensor fusion that plugs into your pipeline.

    Inputs to forward (pooled):
      text_features  : (B, text_in)
      audio_features : (B, audio_in)
      vision_features: (B, vision_in)

    Steps:
      1) Pre-fusion projections -> (B, t_hidden/a_hidden/v_hidden)
      2) Append 1 and do TFN outer products -> (B, fused_dim)
      3) Project fused_dim -> (B, 3*d_model)
      4) SAME fusion_layers MLP head as your other models -> (B, 1)
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
        # if True, audio/vision get TFN-like small MLPs (BN+ReLU); text gets a simple projection
        use_prefusion_mlps: bool = True,
        use_bottleneck_vision: bool=True,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.text_enc = TextEncoder(in_dim=text_in, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=audio_in, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=vision_in,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        self.a_proj = nn.Linear(audio_in, d_model)
        self.v_proj = nn.Linear(vision_in, d_model)
        self.t_proj = nn.Linear(text_in, d_model)       
 
        self.audio_in = int(audio_in)
        self.vision_in = int(vision_in)
        self.text_in = int(text_in)

        self.audio_hidden = int(audio_hidden)
        self.vision_hidden = int(vision_hidden)
        self.text_hidden = int(text_hidden)

        self.dropout_enc = float(dropout_enc)
        self.dropout_head = float(dropout_head)
        self.use_prefusion_mlps = bool(use_prefusion_mlps)

        # -------------------
        # 1) Pre-fusion nets
        # -------------------
        if self.use_prefusion_mlps:
            # TFN-ish for audio/vision (your inputs are already pooled vectors)
            self.audio_subnet = nn.Sequential(
                nn.BatchNorm1d(self.audio_in),
                nn.Dropout(self.dropout_enc),
                nn.Linear(self.audio_in, self.audio_hidden),
                nn.ReLU(),
                nn.Linear(self.audio_hidden, self.audio_hidden),
                nn.ReLU(),
            )
            self.vision_subnet = nn.Sequential(
                nn.BatchNorm1d(self.vision_in),
                nn.Dropout(self.dropout_enc),
                nn.Linear(self.vision_in, self.vision_hidden),
                nn.ReLU(),
                nn.Linear(self.vision_hidden, self.vision_hidden),
                nn.ReLU(),
            )
            # Text: simple projection (since you already encoded it)
            self.text_subnet = nn.Sequential(
                nn.Dropout(self.dropout_enc),
                nn.Linear(self.text_in, self.text_hidden),
            )
        else:
            # Pure linear projections only
            self.audio_subnet = nn.Linear(self.audio_in, self.audio_hidden)
            self.vision_subnet = nn.Linear(self.vision_in, self.vision_hidden)
            self.text_subnet = nn.Linear(self.text_in, self.text_hidden)

        # fused_dim = (Ha+1)(Hv+1)(Ht+1)
        self.fused_dim = (self.audio_hidden + 1) * (self.vision_hidden + 1) * (self.text_hidden + 1)
        # self.fused_dim = (d_model + 1) ** 3

        # -------------------
        # 2) Match your head input dim (3*d_model)
        # -------------------
        # self.proj_to_head = nn.Linear(self.fused_dim, 3 * self.d_model)

        # -------------------
        # 3) SAME regression head structure
        # -------------------
        total_dim = self.fused_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_head),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_head),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _append_one(x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        return torch.cat([ones, x], dim=1)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_length: torch.Tensor, 
        vision_length: torch.Tensor
    ):
        print('Text features:', text_features.shape) 
        print('Audio features:', audio_features.shape)
        print('Vision features:', vision_features.shape)
        # 1) Pre-fusion - Encoding
        t = self.text_enc(text_features)
        a = self.audio_enc(audio_features, audio_length)
        v = self.vision_enc(vision_features, vision_length)
        print('Text features after masked mean pooling:', t.shape)
        print('Audio features after masked mean pooling:', a.shape)
        print('Vision features after masked mean pooling:', v.shape)        

        # after pooling => (B, Fa/Fv/Ft)
#        a = self.a_proj(a)    # (B, d_model)
 #       v = self.v_proj(v)   # (B, d_model)
  #      t = self.t_proj(t)     # (B, d_model)
   #     print('Audio vector after projection:', a.shape)
    #    print('Text vector after projection:', t.shape)
     #   print('Visual vector after projection:', v.shape)

        _a = torch.cat([torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype), a], dim=1)  # (B, d+1)
        _v = torch.cat([torch.ones(v.size(0), 1, device=v.device, dtype=v.dtype), v], dim=1)
        _t = torch.cat([torch.ones(t.size(0), 1, device=t.device, dtype=t.dtype), t], dim=1)


        #t = self.text_subnet(text_features)       # (B, Ht)
        #a = self.audio_subnet(audio_features)     # (B, Ha)
        #v = self.vision_subnet(vision_features)   # (B, Hv)

        # 2) TFN tensor fusion with bias terms
        #_t = self._append_one(t)  # (B, Ht+1)
        # _a = self._append_one(a)  # (B, Ha+1)
        # _v = self._append_one(v)  # (B, Hv+1)

        batch_size = _a.size(0) # B

        # (B, Ha+1, Hv+1)
        fusion_tensor = torch.bmm(_a.unsqueeze(2), _v.unsqueeze(1))

        # (B, (Ha+1)(Hv+1), Ht+1) -> flatten to (batch_size, fused_dim)
        fusion_tensor = fusion_tensor.view(batch_size, -1, 1)
        fusion_tensor = torch.bmm(fusion_tensor, _t.unsqueeze(1)).view(batch_size, -1)

        # 3) Your head
        # head_in = self.proj_to_head(fusion_tensor)          # (B, 3*d_model)
        yhat = self.fusion_layers(fusion_tensor)        # (B, 1)

 #       if return_fused:
#            return yhat, fusion_tensor
        return yhat.squeeze(-1)

