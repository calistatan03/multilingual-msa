import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

# -----------------------
# core metric functions
# -----------------------
def calc_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    print('Labels:', labels)
    print('Predictions:', preds)
    return float(mean_absolute_error(labels, preds))

def calc_correlation(preds: np.ndarray, labels: np.ndarray) -> float:
    # return NaN if undefined (constant arrays)
    if preds.size < 2:
        return float("nan")
    if np.std(preds) < 1e-8 or np.std(labels) < 1e-8:
        return float("nan")
    return float(np.corrcoef(preds, labels)[0, 1])

def calc_binary_accuracy(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.0) -> float:
    binary_preds = (preds > threshold).astype(np.int32)
    binary_labels = (labels > threshold).astype(np.int32)
    return float(accuracy_score(binary_labels, binary_preds))

def calc_binary_f1(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.0) -> float:
    binary_preds = (preds > threshold).astype(np.int32)
    binary_labels = (labels > threshold).astype(np.int32)
    # handle edge case where only one class exists in labels
    return float(f1_score(binary_labels, binary_preds, zero_division=0))


# -----------------------
# predictions collection
# -----------------------
@torch.no_grad()
def get_predictions(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect preds and labels for *multimodal* batches:
      batch["text"], batch["audio"], batch["vision"],
      batch["audio_lengths"], batch["vision_lengths"], batch["y"]
    Model forward must be:
      model(text, audio, vision, audio_lengths, vision_lengths) -> (B,)
    """
    model.eval()

    all_preds = []
    all_labels = []

    for batch in dataloader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        a_len = batch["audio_lengths"].to(device)
        v_len = batch["vision_lengths"].to(device)
        y = batch["y"].to(device)

        yhat = model(text, audio, vision, a_len, v_len)  # (B,)
        all_preds.append(yhat.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return preds, labels


def evaluate(model, dataloader, device, threshold: float = 0.0) -> dict:
    preds, labels = get_predictions(model, dataloader, device)

    if preds.size == 0:
        return {
            "mae": float("nan"),
            "corr": float("nan"),
            "binary_acc": float("nan"),
            "binary_f1": float("nan"),
        }

    return {
        "mae": calc_mae(preds, labels),
        "corr": calc_correlation(preds, labels),
        "binary_acc": calc_binary_accuracy(preds, labels, threshold=threshold),
        "binary_f1": calc_binary_f1(preds, labels, threshold=threshold),
    }

