# experiments/trainer.py

import os
import json
import numpy as np
import torch
import torch.nn as nn

from metrics import evaluate  # expects experiments/metrics.py in same folder


class Trainer:
    """
    Generic trainer for MANY multimodal fusion experiments.

    Assumptions:
      - Batches are dicts with:
          batch["text"]           (B, T_text, 768)
          batch["audio"]          (B, T_audio, 25)
          batch["vision"]         (B, T_vis, 1629)
          batch["audio_lengths"]  (B,)
          batch["vision_lengths"] (B,)
          batch["y"]              (B,)   regression label

      - Model forward signature:
          model(text, audio, vision, audio_lengths, vision_lengths) -> (B,) or (B,1)

      - We train with regression loss (L1 or MSE), but ALSO report:
          MAE, Pearson corr, Binary Acc, Binary F1
        where binary metrics threshold the continuous label at `threshold` (default 0.0).

    Logs:
      - prints one JSON line per eval step
      - appends same JSON to: {out_dir}/metrics_{exp_name}.jsonl

    Saves:
      - best checkpoint by validation MAE to:
          {out_dir}/best_{exp_name}.pt
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        out_dir: str,
        exp_name: str = "experiment",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_type: str = "l1",          # "l1" or "mse"
        grad_clip: float | None = 1.0,
        threshold: float = 0.0,         # for binary acc/f1
        eval_every: int = 1,
        patience: int | None = None,    # early stopping on val MAE
        save_every: int | None = None,  # optional periodic checkpoints (epochs)
    ):
        self.model = model.to(device)
        self.device = device

        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.exp_name = exp_name
        self.best_path = os.path.join(self.out_dir, f"best_{self.exp_name}.pt")
        self.metrics_path = os.path.join(self.out_dir, f"metrics_{self.exp_name}.jsonl")
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_type = loss_type.lower()
        if loss_type == "mse":
            self.crit = nn.MSELoss()
        elif loss_type == "l1":
            self.crit = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_type={loss_type}. Use 'l1' or 'mse'.")

        self.grad_clip = grad_clip
        self.threshold = threshold
        self.eval_every = max(1, int(eval_every))
        self.patience = patience
        self.save_every = save_every

        self.best_val_mae = float("inf")
        self.best_epoch = 0
        self._no_improve = 0

    def _unpack_batch(self, batch: dict):
        text = batch["text"].to(self.device)
        audio = batch["audio"].to(self.device)
        vision = batch["vision"].to(self.device)
        a_len = batch["audio_lengths"].to(self.device)
        v_len = batch["vision_lengths"].to(self.device)
        y = batch["y"].to(self.device)

        # ensure shapes
        if y.dim() > 1:
            y = y.view(-1)

        return text, audio, vision, a_len, v_len, y

    def _forward(self, batch: dict):
        text, audio, vision, a_len, v_len, y = self._unpack_batch(batch)

        yhat = self.model(text, audio, vision, a_len, v_len)
        if isinstance(yhat, (tuple, list)):
            # in case some models return (pred, aux)
            yhat = yhat[0]

        # squeeze to (B,)
        if yhat.dim() > 1:
            yhat = yhat.squeeze(-1)

        return yhat, y

    def train_one_epoch(self, train_loader):
        self.model.train()
        losses = []

        for step, batch in enumerate(train_loader, start=1):
            yhat, y = self._forward(batch)
            loss = self.crit(yhat, y)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.opt.step()
            losses.append(loss.item())
            if step % 50 == 0: 
                print(f"[train] step={step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def run_eval(self, loader):
        # uses experiments/metrics.py
        return evaluate(self.model, loader, self.device, threshold=self.threshold)

    def _save_best(self, epoch: int, val_metrics: dict):
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "best_val_mae": self.best_val_mae,
                "val_metrics": val_metrics,
                "exp_name": self.exp_name,
            },
            self.best_path,
        )

    def _save_checkpoint(self, epoch: int, tag: str = ""):
        name = f"epoch_{epoch}.pt" if not tag else f"epoch_{epoch}_{tag}.pt"
        path = os.path.join(self.ckpt_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "best_val_mae": self.best_val_mae,
                "exp_name": self.exp_name,
            },
            path,
        )

    def fit(self, train_loader, val_loader, epochs: int = 30):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)

            # periodic checkpoint (optional)
            if self.save_every is not None and epoch % int(self.save_every) == 0:
                self._save_checkpoint(epoch)

            # evaluate
            if epoch % self.eval_every == 0:
                val_metrics = self.run_eval(val_loader)

                rec = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mae": val_metrics["mae"],
                    "val_corr": val_metrics["corr"],
                    "val_binary_acc": val_metrics["binary_acc"],
                    "val_binary_f1": val_metrics["binary_f1"],
                    "threshold": self.threshold,
                }

                # print + append jsonl
                print(json.dumps(rec))
                with open(self.metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

                # best by val MAE
                if val_metrics["mae"] < self.best_val_mae:
                    self.best_val_mae = val_metrics["mae"]
                    self.best_epoch = epoch
                    self._no_improve = 0
                    self._save_best(epoch, val_metrics)
                else:
                    self._no_improve += 1

                # early stopping
                if self.patience is not None and self._no_improve >= int(self.patience):
                    stop_rec = {"early_stop": True, "stopped_epoch": epoch, "best_epoch": self.best_epoch}
                    print(json.dumps(stop_rec))
                    with open(self.metrics_path, "a") as f:
                        f.write(json.dumps(stop_rec) + "\n")
                    break

        return self.best_path

    def load_best(self) -> bool:
        if not os.path.exists(self.best_path):
            return False
        ckpt = torch.load(self.best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        return True

    # add near the bottom of Trainer (e.g., above test())

    @torch.no_grad()
    def save_predictions(self, loader, save_path: str, split: str = "test"):
        """
        Saves per-example predictions for clustering/error analysis.

        Expected batch keys (optional for id/text):
          - "id" or "ids"
          - "raw_text" or "raw_texts"
        Always uses:
          - "y" and model output

        Output JSON format:
          {
            "split": "test",
            "exp_name": "...",
            "threshold": 0.0,
            "ids": [...],
            "raw_texts": [...],
            "y_true": [...],
            "yhat": [...]
          }
        """
        self.model.eval()

        ids = []
        texts = []
        y_true_all = []
        yhat_all = []

        for batch in loader:
            # forward pass
            yhat, y = self._forward(batch)  # both (B,)
            y_true_all.append(y.detach().cpu())
            yhat_all.append(yhat.detach().cpu())

            # optional metadata
            if "id" in batch:
                bid = batch["id"]
                if torch.is_tensor(bid):
                    bid = bid.detach().cpu().tolist()
                ids.extend(list(bid))
            elif "ids" in batch:
                bid = batch["ids"]
                if torch.is_tensor(bid):
                    bid = bid.detach().cpu().tolist()
                ids.extend(list(bid))

            if "raw_text" in batch:
                bt = batch["raw_text"]
                if torch.is_tensor(bt):
                    bt = bt.detach().cpu().tolist()
                texts.extend(list(bt))
            elif "raw_texts" in batch:
                bt = batch["raw_texts"]
                if torch.is_tensor(bt):
                    bt = bt.detach().cpu().tolist()
                texts.extend(list(bt))

        y_true = torch.cat(y_true_all).numpy().tolist()
        yhat = torch.cat(yhat_all).numpy().tolist()

        n = len(y_true)
        # fallbacks if ids/text not provided
        if len(ids) != n:
            ids = list(range(n))
        if len(texts) != n:
            texts = [""] * n

        obj = {
            "split": split,
            "exp_name": self.exp_name,
            "threshold": self.threshold,
            "ids": ids,
            "raw_texts": texts,
            "y_true": y_true,
            "yhat": yhat,
        }

        with open(save_path, "w") as f:
            json.dump(obj, f)

    @torch.no_grad()
    def test(self, test_loader):
        """
        Convenience method: loads best (if exists) and evaluates on test_loader.
        """
        self.load_best()
        test_metrics = self.run_eval(test_loader)
        rec = {
            "split": "test",
            "best_epoch": self.best_epoch,
            "best_val_mae": self.best_val_mae,
            "test_mae": test_metrics["mae"],
            "test_corr": test_metrics["corr"],
            "test_binary_acc": test_metrics["binary_acc"],
            "test_binary_f1": test_metrics["binary_f1"],
            "threshold": self.threshold,
        }
        print(json.dumps(rec))
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return test_metrics

