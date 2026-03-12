import json
import numpy as np
import pandas as pd

def load_preds(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    df = pd.DataFrame({
        "id": [str(x) for x in d["ids"]],
        "raw_text": ["" if x is None else str(x) for x in d["raw_texts"]],
        "y_true": d["y_true"],
        "yhat": d["yhat"],
    })
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
    df = df.dropna(subset=["y_true", "yhat"]).reset_index(drop=True)
    return df

def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based 5-class grouping on y_true:

      negative         : y <= -0.8
      weakly_negative  : -0.8 < y < 0
      neutral          : y == 0 (within eps tolerance)
      weakly_positive  : 0 < y < 0.8
      positive         : y >= 0.8

    This avoids 'other' for continuous values like 0.1, 0.3, 0.7, etc.
    """
    df = df.copy()
    y = pd.to_numeric(df["y_true"], errors="coerce")

    eps = 1e-8
    cls = np.full(len(df), "other", dtype=object)

    cls[y <= -0.8] = "negative"
    cls[(y > -0.8) & (y < -eps)] = "weakly_negative"
    cls[np.isclose(y, 0.0, atol=1e-6)] = "neutral"
    cls[(y > eps) & (y < 0.8)] = "weakly_positive"
    cls[y >= 0.8] = "positive"

    df["sentiment_class"] = cls
    return df

def bin_metrics(df: pd.DataFrame, pred_col: str, threshold: float = 0.0) -> pd.DataFrame:
    """
    Computes MAE and simple binary acc/f1 by bins.
    """
    out_rows = []
    for b, g in df.groupby("sentiment_class", dropna=False):
        y = g["y_true"].to_numpy()
        yhat = g[pred_col].to_numpy()
        mae = float(np.mean(np.abs(yhat - y)))

        # binary metrics (threshold at 0 by default)
        yb = (y >= threshold).astype(int)
        ph = (yhat >= threshold).astype(int)
        acc = float(np.mean(yb == ph))

        # F1
        tp = int(np.sum((ph == 1) & (yb == 1)))
        fp = int(np.sum((ph == 1) & (yb == 0)))
        fn = int(np.sum((ph == 0) & (yb == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        out_rows.append({
            "bin": str(b),
            "n": int(len(g)),
            "mae": mae,
            "binary_acc": acc,
            "binary_f1": float(f1),
            "y_true_mean": float(np.mean(y)),
            "abs_y_mean": float(np.mean(np.abs(y))),
        })

    return pd.DataFrame(out_rows).sort_values("bin")

def main(
    audio_json: str,
    vision_json: str,
    out_prefix: str,
    top_k: int = 200,
):
    # Load both
    a = load_preds(audio_json).rename(columns={"yhat": "yhat_audio"})
    v = load_preds(vision_json).rename(columns={"yhat": "yhat_vision"})

    # Align by id (inner join)
    df = a.merge(v[["id", "yhat_vision"]], on="id", how="inner")

    # Add bins + errors
    df = add_bins(df)
    df["abs_err_audio"] = (df["yhat_audio"] - df["y_true"]).abs()
    df["abs_err_vision"] = (df["yhat_vision"] - df["y_true"]).abs()
    df["delta_audio_better"] = df["abs_err_vision"] - df["abs_err_audio"]  # >0 means audio better

    # --------------------------
    # (2) Intensity bin analysis
    # --------------------------
    audio_bin = bin_metrics(df, pred_col="yhat_audio")
    vision_bin = bin_metrics(df, pred_col="yhat_vision")

    audio_bin.to_csv(f"{out_prefix}_bin_metrics_audio.csv", index=False)
    vision_bin.to_csv(f"{out_prefix}_bin_metrics_vision.csv", index=False)

    # Combine for side-by-side comparison
    comb = audio_bin.merge(
        vision_bin,
        on="bin",
        suffixes=("_audio", "_vision"),
        how="outer"
    )
    comb.to_csv(f"{out_prefix}_bin_metrics_side_by_side.csv", index=False)

    # --------------------------
    # (3) “Audio wins” examples
    # --------------------------
    # Audio wins: largest positive delta
    audio_wins = df.sort_values("delta_audio_better", ascending=False).head(top_k)
    vision_wins = df.sort_values("delta_audio_better", ascending=True).head(top_k)

    keep_cols = [
        "id", "raw_text", "y_true", "sentiment_class",
        "yhat_audio", "yhat_vision",
        "abs_err_audio", "abs_err_vision",
        "delta_audio_better",
    ]

    audio_wins[keep_cols].to_csv(f"{out_prefix}_top_audio_wins.csv", index=False)
    vision_wins[keep_cols].to_csv(f"{out_prefix}_top_vision_wins.csv", index=False)

    # --------------------------
    # (5) Audio win rate by bin (+ avg abs error gap)
    # --------------------------
    df["audio_beats_vision"] = (df["delta_audio_better"] > 0).astype(int)

    # NEW: absolute gap in errors regardless of which model wins
    df["abs_err_gap"] = (df["abs_err_vision"] - df["abs_err_audio"]).abs()

    win_rate = (
        df.groupby("sentiment_class")
          .agg(
              n=("id", "count"),
              audio_win_rate=("audio_beats_vision", "mean"),
              avg_abs_err_gap=("abs_err_gap", "mean"),              # NEW
              avg_signed_err_gap=("delta_audio_better", "mean"),    # OPTIONAL but helpful
          )
          .reset_index()
    )
    win_rate.to_csv(f"{out_prefix}_audio_win_rate_by_bin.csv", index=False)

    # Save merged aligned file (useful for further analysis)
    df.to_csv(f"{out_prefix}_aligned_audio_vs_vision.csv", index=False)

    print("[done] wrote:")
    print(f"  {out_prefix}_bin_metrics_audio.csv")
    print(f"  {out_prefix}_bin_metrics_vision.csv")
    print(f"  {out_prefix}_bin_metrics_side_by_side.csv")
    print(f"  {out_prefix}_audio_win_rate_by_bin.csv")
    print(f"  {out_prefix}_top_audio_wins.csv")
    print(f"  {out_prefix}_top_vision_wins.csv")
    print(f"  {out_prefix}_aligned_audio_vs_vision.csv")

if __name__ == "__main__":
    # -------- EDIT THESE PATHS --------
    AUDIO_JSON = "/hpctmp/scratch/e0968015/mosei/FusionRuns/audio_guided_attn_run3/preds_test.json"
    VISION_JSON = "/hpctmp/scratch/e0968015/mosei/FusionRuns/vision_guided_attn_run3/preds_test.json"
    OUT_PREFIX = "/hpctmp/scratch/e0968015/mosei/FusionRuns/posthoc_audio_vs_vision_run3/mosei_A35"

    import os
    os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)

    main(AUDIO_JSON, VISION_JSON, OUT_PREFIX, top_k=200)
