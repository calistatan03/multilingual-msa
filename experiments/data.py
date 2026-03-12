import pickle
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

class MOSEIDataset(Dataset): 
    """
    Expects a split dict like data["train"] / data["valid"] / data["test"] from your features.pkl.

    Required keys in split_dict:
      - "text":           (N, T_text, 768)  float
      - "audio":          (N, T_audio, 25)  float
      - "vision":         (N, T_vis, 1629)  float
      - "audio_lengths":  (N,)              int
      - "vision_lengths": (N,)              int
      - "regression_labels": (N,) or (N,1)  float
      - "id": (N,)
      - "raw_texts": (N,)

    Returns a dict batch item with keys expected by Trainer:
      "text","audio","vision","audio_lengths","vision_lengths","y"
    """
    def __init__(self, split_dict: dict, label_key: str = "regression_label"): 
        self.text = np.asarray(split_dict["text"], dtype=np.float32)
        self.audio = np.asarray(split_dict["audio"], dtype=np.float32)
        self.vision = np.asarray(split_dict["vision"], dtype=np.float32)
        self.id = np.asarray(split_dict["id"], dtype=np.str_)
        self.raw_texts = np.asarray(split_dict["raw_text"], dtype=np.str_)

        self.audio_len = np.asarray(split_dict["audio_lengths"], dtype=np.int64)
        self.vision_len = np.asarray(split_dict["vision_lengths"], dtype=np.int64)

        y = np.asarray(split_dict[label_key], dtype=np.float32).reshape(-1)
        # self.y = y
        y_min = float(y.min())
        y_max = float(y.max())
        eps = 1e-4
        if abs(y_min + 2.0) < eps and abs(y_max - 2.0) < eps:
            y = y / 2.0

        self.y = y

        n = self.y.shape[0]
        assert self.text.shape[0] == n, f"text N mismatch: {self.text.shape[0]} vs {n}"
        assert self.audio.shape[0] == n, f"audio N mismatch: {self.audio.shape[0]} vs {n}"
        assert self.vision.shape[0] == n, f"vision N mismatch: {self.vision.shape[0]} vs {n}"
        assert self.audio_len.shape[0] == n, f"audio_lengths N mismatch: {self.audio_len.shape[0]} vs {n}"
        assert self.vision_len.shape[0] == n, f"vision_lengths N mismatch: {self.vision_len.shape[0]} vs {n}"

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        text_i = torch.as_tensor(self.text[idx], dtype=torch.float32)  # (T_text, 768)
        # infer length: count tokens whose embedding is not all zeros
        # (treat token as valid if L1 norm > 0)
        valid = (text_i.abs().sum(dim=-1) > 0)
        text_len = valid.long().sum()

        return {
            "text": torch.as_tensor(self.text[idx], dtype=torch.float32),
            "audio": torch.as_tensor(self.audio[idx], dtype=torch.float32),               # (T_audio, 25)
            "vision": torch.as_tensor(self.vision[idx], dtype=torch.float32),             # (T_vis, 1629)
            "audio_lengths": torch.tensor(self.audio_len[idx], dtype=torch.long),
            "vision_lengths": torch.tensor(self.vision_len[idx], dtype=torch.long),
            "text_lengths": text_len,
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "id": str(self.id[idx]),
            "raw_text": str(self.raw_texts[idx])
        }

def load_splits_from_pkl(features_pkl: str):
    """
    Loads the full dict with 'train'/'valid'/'test' from features.pkl.
    """
    with open(features_pkl, "rb") as f:
        data = pickle.load(f)
    return data


def make_loaders(
    features_pkl: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    label_key: str = "regression_labels",
    use_test_if_missing: bool = False,
):
    """
    Returns: train_loader, valid_loader, test_loader
    If 'test' split is missing:
      - if use_test_if_missing=True -> uses 'valid' as test
      - else -> returns None for test_loader
    """
    data = load_splits_from_pkl(features_pkl)

    train_ds = MOSEIDataset(data["train"], label_key=label_key)
    valid_ds = MOSEIDataset(data["valid"], label_key=label_key)

    test_loader = None
    if "test" in data:
        test_ds = MOSEIDataset(data["test"], label_key=label_key)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif use_test_if_missing:
        test_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,   # keep 0 on PBS unless you're sure
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader
