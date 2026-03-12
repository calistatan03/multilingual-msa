# experiments/data_time_align.py
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils_align import resample_to_len

class AlignedTimeDataset(Dataset):
    def __init__(self, split_dict: dict, label_key: str = "regression_labels"):
        self.text = np.asarray(split_dict["text"], dtype=np.float32)     # (N, Tt, 768)
        self.audio = np.asarray(split_dict["audio"], dtype=np.float32)   # (N, Ta, 25)
        self.vision = np.asarray(split_dict["vision"], dtype=np.float32) # (N, Tv, 1629)

        self.text_len = np.full((self.text.shape[0],), self.text.shape[1], dtype=np.int64)  # if no text lengths
        self.audio_len = np.asarray(split_dict["audio_lengths"], dtype=np.int64)
        self.vision_len = np.asarray(split_dict["vision_lengths"], dtype=np.int64)

        y = np.asarray(split_dict[label_key], dtype=np.float32).reshape(-1)
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "text": torch.from_numpy(self.text[idx]),      # (Tt, 768)
            "audio": torch.from_numpy(self.audio[idx]),    # (Ta, 25)
            "vision": torch.from_numpy(self.vision[idx]),  # (Tv, 1629)
            "text_len": int(self.text_len[idx]),
            "audio_len": int(self.audio_len[idx]),
            "vision_len": int(self.vision_len[idx]),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
        }

def collate_align_to_audio(batch):
    # Align each sample to its own audio_len, then pad within batch
    aligned = []
    max_T = 1

    for b in batch:
        tgt_len = max(1, int(b["audio_len"]))

        t = resample_to_len(b["text"],   b["text_len"],   tgt_len)   # (tgt_len, 768)
        a = resample_to_len(b["audio"],  b["audio_len"],  tgt_len)   # (tgt_len, 25)
        v = resample_to_len(b["vision"], b["vision_len"], tgt_len)   # (tgt_len, 1629)

        mask = torch.ones(tgt_len, dtype=torch.bool)

        aligned.append((t, a, v, mask, b["label"]))
        max_T = max(max_T, tgt_len)

    # Pad to max_T
    B = len(batch)
    text = torch.zeros(B, max_T, 768)
    audio = torch.zeros(B, max_T, 25)
    vision = torch.zeros(B, max_T, 1629)
    mask = torch.zeros(B, max_T, dtype=torch.bool)
    labels = torch.stack([x[-1] for x in aligned], dim=0)

    for i, (t, a, v, m, _) in enumerate(aligned):
        T = t.shape[0]
        text[i, :T] = t
        audio[i, :T] = a
        vision[i, :T] = v
        mask[i, :T] = m

    return {"text": text, "audio": audio, "vision": vision, "mask": mask, "label": labels}

