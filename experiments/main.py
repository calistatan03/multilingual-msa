#!/usr/bin/env python3
# experiments/main.py

import os
import argparse
import json
import numpy as np
import torch

from data import make_loaders
from trainer import Trainer

# Your fusion model + encoders wiring
from early_fusion import EarlyFusionModel
from attention import AttnFusionModel
from coattn_fusion import CoAttentionFusionSystem
from text_guided_attn import TextGuidedAttnSystem
from audio_guided_attn import AudioGuidedAttnSystem
from vision_guided_attn import VisionGuidedAttnSystem
from trimodal_attention import TriFANLikeSystem
from tensor_fusion import TensorFusionSystem
from graph_fusion import GraphFusion
from kernel_fusion import KernelFusionSystem
from bibimodal_fusion import BiBimodalFusionSystem
from lowrank_fusion import LowRankFusionSystem
from encoders.text import TextEncoder
from encoders.audio import AudioEncoder
from encoders.vision import VisionEncoder
import torch.nn as nn


class EarlyFusionSystem(nn.Module):
    """
    Encoders -> utterance vectors -> EarlyFusionModel head (feature-level concatenation).
    Assumes:
      TextEncoder: returns (B, d_model)
      AudioEncoder: forward(x, lengths) -> (B, d_model)
      VisionEncoder: forward(x, lengths) -> (B, d_model)
    """
    def __init__(
        self,
        d_model=256,
        hidden_dim=256,
        dropout_enc=0.1,
        dropout_head=0.3,
        audio_kernel=3,
        vision_kernel=3,
        use_bottleneck_vision=True,
    ):
        super().__init__()

        self.text_enc = TextEncoder(in_dim=768, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=35, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=1629,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        self.head = EarlyFusionModel(
            text_dim=d_model,
            audio_dim=d_model,
            visual_dim=d_model,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_head,
        )

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        z_t = self.text_enc(text)                      # (B, d_model)
        z_a = self.audio_enc(audio, audio_lengths)     # (B, d_model)
        z_v = self.vision_enc(vision, vision_lengths)  # (B, d_model)
        yhat = self.head(z_t, z_a, z_v).squeeze(-1)    # (B,)
        return yhat

class AttnFusionSystem(nn.Module):
    def __init__(
        self,
        d_model=256,
        hidden_dim=256,
        dropout_enc=0.1,
        dropout_head=0.3,
        audio_kernel=3,
        vision_kernel=3,
        use_bottleneck_vision=True,
        log_alphas=False,
    ):
        super().__init__()
        self.log_alphas = log_alphas

        self.text_enc = TextEncoder(in_dim=768, out_dim=d_model, dropout=dropout_enc)
        self.audio_enc = AudioEncoder(in_dim=35, d_model=d_model, kernel_size=audio_kernel, dropout=dropout_enc)
        self.vision_enc = VisionEncoder(
            in_dim=1629,
            d_model=d_model,
            kernel_size=vision_kernel,
            dropout=dropout_enc,
            use_bottleneck=use_bottleneck_vision,
        )

        self.head = AttnFusionModel(d_model=d_model, hidden_dim=hidden_dim, dropout_rate=dropout_head)

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        z_t = self.text_enc(text)
        z_a = self.audio_enc(audio, audio_lengths)
        z_v = self.vision_enc(vision, vision_lengths)

        if self.log_alphas:
            yhat, alphas = self.head(z_t, z_a, z_v, return_alphas=True)
            return yhat.squeeze(-1), alphas
        else:
            yhat = self.head(z_t, z_a, z_v)
            return yhat.squeeze(-1)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()

    # I/O
    ap.add_argument("--features_pkl", required=True, help="Path to features.pkl")
    ap.add_argument("--out_dir", required=True, help="Where to write logs + checkpoints")
    ap.add_argument("--exp_name", default="early_fusion", help="Name used for best_*.pt + metrics_*.jsonl")

    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--loss_type", default="l1", choices=["l1", "mse"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.0)

    # dataloader
    ap.add_argument("--num_workers", type=int, default=0)  # keep 0 on PBS unless safe
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--use_valid_as_test", action="store_true")

    # model/encoder
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout_enc", type=float, default=0.1)
    ap.add_argument("--dropout_head", type=float, default=0.3)
    ap.add_argument("--audio_kernel", type=int, default=3)
    ap.add_argument("--vision_kernel", type=int, default=3)
    ap.add_argument("--no_vision_bottleneck", action="store_true")
    ap.add_argument("--fusion", default="coattn", choices=["early_fusion", "attn", "coattn", "text_guided_attn", "trimodal_attn", "tensor_fusion", "graph_fusion", "kernel_fusion", "bibimodal_fusion", "lowrank_fusion", "audio_guided_attn", "vision_guided_attn"])
    ap.add_argument("--log_alphas", action="store_true")


    # misc
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = make_loaders(
        features_pkl=args.features_pkl,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        label_key="regression_labels",
        use_test_if_missing=args.use_valid_as_test,
    )
    print('Data loaders initialised')

    # Model
    if args.fusion == "early_fusion":
        model = EarlyFusionSystem(
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout_enc=args.dropout_enc,
        dropout_head=args.dropout_head,
        audio_kernel=args.audio_kernel,
        vision_kernel=args.vision_kernel,
        use_bottleneck_vision=(not args.no_vision_bottleneck),
    )
    elif args.fusion == "attn":
        model = AttnFusionSystem(
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout_enc=args.dropout_enc,
        dropout_head=args.dropout_head,
        audio_kernel=args.audio_kernel,
        vision_kernel=args.vision_kernel,
        use_bottleneck_vision=(not args.no_vision_bottleneck),
    )
    elif args.fusion == "coattn": 
        model = CoAttentionFusionSystem(
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
    )
    elif args.fusion == "text_guided_attn": 
        model = TextGuidedAttnSystem(
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout_enc=args.dropout_enc,
        dropout_head=args.dropout_head,
        audio_kernel=args.audio_kernel,
        vision_kernel=args.vision_kernel,
        use_bottleneck_vision=(not args.no_vision_bottleneck),
    )
    elif args.fusion == "trimodal_attn": 
        model = TriFANLikeSystem(
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout_enc=args.dropout_enc,
        dropout_head=args.dropout_head,
        audio_in=35,
        vision_in=1629,
        text_in=768,
        audio_kernel=args.audio_kernel,
        vision_kernel=args.vision_kernel,
        use_bottleneck_vision=(not args.no_vision_bottleneck),
        n_heads=4,
    )
    elif args.fusion == "tensor_fusion": 
        model = TensorFusionSystem(
        d_model=16,
        hidden_dim=args.hidden_dim,
        dropout_enc=args.dropout_enc,
        dropout_head=args.dropout_head,
        audio_in=16,
        vision_in=1629,
        text_in=768,
        audio_kernel=args.audio_kernel,
        vision_kernel=args.vision_kernel,
        # TFN pre-fusion sizes (start small)
        audio_hidden=16,
        vision_hidden=16,
        text_hidden=16,
        use_prefusion_mlps=True,
    )      
    elif args.fusion == "graph_fusion": 
        model = GraphFusion(
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout_enc=args.dropout_enc,
        dropout_head=args.dropout_head,
        audio_in=16,
        vision_in=1629,
        text_in=768,
        audio_kernel=args.audio_kernel,
        vision_kernel=args.vision_kernel,
        audio_hidden=16,
        vision_hidden=16,
        text_hidden=16,
        use_prefusion_mlps=True,
    )
    elif args.fusion == "kernel_fusion": 
        model = KernelFusionSystem(
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            rff_dim=512,
            gamma_text=1.0, gamma_audio=1.0, gamma_vision=1.0,
            wt=1/3, wa=1/3, wv=1/3,
        )
    elif args.fusion == "bibimodal_fusion": 
        model = BiBimodalFusionSystem(
        d_model = args.d_model, 
        hidden_dim=args.hidden_dim
    )
    elif args.fusion == "lowrank_fusion": 
        model = LowRankFusionSystem( 
        d_model=args.d_model, 
        hidden_dim=args.hidden_dim, 
        dropout_enc=args.dropout_enc, 
        dropout_head=args.dropout_head 
    )
    elif args.fusion == "audio_guided_attn": 
        model = AudioGuidedAttnSystem( 
        d_model=args.d_model, 
        hidden_dim=args.hidden_dim
    )
    else: 
        model = VisionGuidedAttnSystem( 
        d_model=args.d_model, 
        hidden_dim=args.hidden_dim
    )
        
    print('Model initialised')


    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        out_dir=args.out_dir,
        exp_name=args.exp_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        grad_clip=args.grad_clip,
        threshold=args.threshold,
        eval_every=args.eval_every,
        patience=args.patience,
        save_every=None,
    )
    print('Trainer initialised')

    best_path = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # Test (if available)
    results = {"best_ckpt": best_path}
    if test_loader is not None:
        test_metrics = trainer.test(test_loader)
        results["test"] = test_metrics
        preds_path = os.path.join(args.out_dir, "preds_test.json")
        trainer.save_predictions(test_loader, preds_path, split="test")
        print(f"Predictions saved at: {preds_path}")

    # Save final summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"done": True, **results}))


if __name__ == "__main__":
    main()

