# experiments/utils_align.py
import torch
import torch.nn.functional as F

def resample_to_len(x: torch.Tensor, src_len: int, tgt_len: int) -> torch.Tensor:
    """
    x: (T, D) float tensor, where only first src_len timesteps are valid
    returns: (tgt_len, D)
    """
    assert x.dim() == 2
    T, D = x.shape
    src_len = int(max(1, min(src_len, T)))
    tgt_len = int(max(1, tgt_len))

    x = x[:src_len]                       # (src_len, D)
    x = x.transpose(0, 1).unsqueeze(0)    # (1, D, src_len)
    y = F.interpolate(x, size=tgt_len, mode="linear", align_corners=False)
    y = y.squeeze(0).transpose(0, 1)      # (tgt_len, D)
    return y

