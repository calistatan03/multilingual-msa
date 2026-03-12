# dataset.py
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle(path: str) -> Dict[str, Any]:
    # Handles python2 pickles too
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


@dataclass
class Batch:
    x: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor


class UnimodalPickleDataset(Dataset):
    """
    Expects data structure:
    data[split]['audio'] -> (N, Ta, da) float
    data[split]['vision'] -> (N, Tv, dv) float
    data[split]['text'] -> (N, Tl, 768) float
    data[split]['audio_lengths'] -> (N,) int
    data[split]['vision_lengths'] -> (N,) int
    data[split]['regression_labels'] -> (N,) float or (N,1)
    """

    def __init__(
        self,
        pkl_path: str,
        split: str,
        modality: str,
        label_key: str = "regression_labels",
    ):
        super().__init__()
        assert modality in {"text", "audio", "vision"}, "modality must be text/audio/vision"
        assert split in {"train", "valid", "test"}, "split must be train/valid/test"

        self.data = load_pickle(pkl_path)
        if split not in self.data:
            raise KeyError(f"Split '{split}' not found in {pkl_path}. Found: {list(self.data.keys())}")

        self.split = split
        self.modality = modality
        self.label_key = label_key

        split_data = self.data[split]

        # Choose feature keys
        if modality == "text":
            self.x_key = "text"
            self.len_key = None   # infer from zero-padded vectors
        elif modality == "audio":
            self.x_key = "audio"
            self.len_key = "audio_lengths"
        else:
            self.x_key = "vision"
            self.len_key = "vision_lengths"

        if self.x_key not in split_data:
            raise KeyError(f"Feature key '{self.x_key}' not in data['{split}']. Keys: {list(split_data.keys())}")

        if label_key not in split_data:
            raise KeyError(f"Label key '{label_key}' not in data['{split}']. Keys: {list(split_data.keys())}")

        self.X = np.asarray(split_data[self.x_key])
        self.Y = np.asarray(split_data[label_key]).reshape(-1)

        if self.len_key is None:
            # Infer text lengths from zero-padded rows
            # Assumes padded rows are all zeros
            valid = np.abs(self.X).sum(axis=-1) > 0   # [N, T]
            self.lengths = valid.sum(axis=-1).astype(np.int64)
        else:
            if self.len_key not in split_data:
                raise KeyError(f"Length key '{self.len_key}' not in data['{split}'].")
            self.lengths = np.asarray(split_data[self.len_key]).astype(np.int64)
            
            # Fix bad provided lengths for vision/audio if needed
            bad_idx = np.where(self.lengths <= 0)[0]
            if len(bad_idx) > 0:
                print(f"[warn] Found {len(bad_idx)} non-positive lengths for modality '{modality}'. Recomputing them from X.shape[0].")
            for i in bad_idx:
                self.lengths[i] = self.X[i].shape[0]

        if len(self.X) != len(self.Y):
            raise ValueError(f"X and Y length mismatch: {len(self.X)} vs {len(self.Y)}")
        if len(self.lengths) != len(self.X):
            raise ValueError(f"lengths and X length mismatch: {len(self.lengths)} vs {len(self.X)}")

        if np.any(self.lengths <= 0):
            bad_idx = np.where(self.lengths <= 0)[0]
            if len(bad_idx) > 0:
                print(f"[debug] modality={modality}")
                print(f"[debug] number of bad lengths: {len(bad_idx)}")
                print(f"[debug] first 10 bad indices: {bad_idx[:10]}")
                print(f"[debug] bad length values: {self.lengths[bad_idx[:10]]}")
                if len(bad_idx) > 0:
                    i = bad_idx[0]
                    print(f"[debug] X[{i}].shape = {self.X[i].shape}")
            raise ValueError(
                f"Some inferred/provided lengths are <= 0 for modality '{modality}'. "
                f"Check the stored features and padding."
            )

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor(np.asarray(self.X[idx], dtype=np.float32))
        l = torch.tensor(self.lengths[idx], dtype=torch.long)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, l, y


def collate_pad(batch):
    """
    Pads sequences to max length in batch.
    batch: list of (x: [T,D], len: [], y: [])
    returns Batch(x: [B,Tmax,D], lengths: [B], y:[B])
    """
    xs, ls, ys = zip(*batch)
    lengths = torch.stack(ls, dim=0)
    y = torch.stack(ys, dim=0)

    max_len = int(lengths.max().item())
    feat_dim = xs[0].shape[-1]
    B = len(xs)

    x_padded = torch.zeros((B, max_len, feat_dim), dtype=torch.float32)

    for i, (x, l) in enumerate(zip(xs, ls)):
        t = min(int(l.item()), x.shape[0], max_len)
        x_padded[i, :t, :] = x[:t, :]

    lengths = lengths.clamp(min=1, max=max_len)

    return Batch(x=x_padded, lengths=lengths, y=y)
