# train_unimodal.py
import math
import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict
from models import (
    TextEncoder,
    AVEncoder,
    UnimodalRegressor,
    build_unimodal_model
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UnimodalPickleDataset, collate_pad


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    l1 = nn.L1Loss(reduction="sum")

    total_loss = 0.0
    total_n = 0
    preds_all = []
    ys_all = []
 
    for batch in loader:
        x = batch.x.to(device)
        lengths = batch.lengths.to(device)
        y = batch.y.to(device)

        out = model(x, lengths)
        yhat = out["logits"]

        loss = l1(yhat, y)

        total_loss += float(loss.item())
        total_n += y.numel()

        preds_all.append(yhat.detach().cpu())
        ys_all.append(y.detach().cpu())

    mae = total_loss / max(total_n, 1)
    preds = torch.cat(preds_all, dim=0)
    ys = torch.cat(ys_all, dim=0)

    # Pearson correlation (optional but useful)
    vx = preds - preds.mean()
    vy = ys - ys.mean()
    denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt()).clamp(min=1e-8)
    corr = float((vx * vy).sum().item() / denom.item())

    return {"mae": mae, "corr": corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Path to features.pkl")
    ap.add_argument("--dataset_name", default="dataset", help="For logging only (mosei/chsims etc.)")
    ap.add_argument("--modality", required=True, choices=["text", "audio", "vision"])
    ap.add_argument("--label_key", default="regression_labels",
                    help="regression_labels or regression_labels_T/A/V")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--conv_kernel", type=int, default=3)
    ap.add_argument("--max_seq_len", type=int, default=5000)

    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = UnimodalPickleDataset(args.pkl, "train", args.modality, args.label_key)
    valid_ds = UnimodalPickleDataset(args.pkl, "valid", args.modality, args.label_key)
    test_ds  = UnimodalPickleDataset(args.pkl, "test",  args.modality, args.label_key)

    # Infer input dim
    x0, _, _ = train_ds[0]
    in_dim = x0.shape[-1]

    if args.modality == "text":
        encoder = TextEncoder(
                in_dim=in_dim,           # should be 768
                d_model=args.d_model,
                conv_kernel=1,           # recommended for pre-extracted BERT features
                dropout=args.dropout,
        )
    else:
        encoder = AVEncoder(
                in_dim=in_dim,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                dropout=args.dropout,
                conv_kernel=args.conv_kernel,
                max_seq_len=args.max_seq_len,
        )

    model = UnimodalRegressor(encoder=encoder, d_model=args.d_model).to(device)    

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_pad
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_pad
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_pad
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss(reduction="mean")  # MAE

    best_mae = float("inf")
    best_path = outdir / f"best_{args.dataset_name}_{args.modality}.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch.x.to(device)
            lengths = batch.lengths.to(device)
            y = batch.y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths)
            yhat = out["logits"]
            loss = loss_fn(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        train_mae = running / max(n_batches, 1)
        val_metrics = evaluate(model, valid_loader, device)

        row = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_metrics["mae"],
            "val_corr": val_metrics["corr"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "in_dim": in_dim,
                    "best_val_mae": best_mae,
                },
                best_path,
            )

    # Final test evaluation with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)

    results = {
        "dataset": args.dataset_name,
        "modality": args.modality,
        "label_key": args.label_key,
        "best_val_mae": best_mae,
        "test_mae": test_metrics["mae"],
        "test_corr": test_metrics["corr"],
        "checkpoint": str(best_path),
        "history": history,
    }

    with open(outdir / f"metrics_{args.dataset_name}_{args.modality}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("DONE:", json.dumps({k: results[k] for k in ["dataset", "modality", "best_val_mae", "test_mae", "test_corr"]}, indent=2))


if __name__ == "__main__":
    main()

