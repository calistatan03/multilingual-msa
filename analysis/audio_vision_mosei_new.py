#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import librosa


# --------------------------
# IO helpers
# --------------------------
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


def parse_id(id_str: str):
    s = str(id_str)
    if "$_$" not in s:
        return None, None
    vid, clip = s.split("$_$", 1)
    return vid, clip

def video_path_from_id(raw_dir: str, videoid: str, clipid: str) -> str:
    return str(Path("/hpctmp") / raw_dir.lstrip("/") / videoid / f"{clipid}.mp4")

def try_extract_wav_with_ffmpeg(video_path: str, wav_path: str, sr: int = 16000) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def load_audio_from_mp4(mp4_path: str, sr: int = 16000) -> np.ndarray :
    tmp_wav = str(Path(mp4_path).with_suffix(f".tmp{sr}.wav"))
    if try_extract_wav_with_ffmpeg(mp4_path, tmp_wav, sr=sr) and os.path.exists(tmp_wav):
        y, _ = librosa.load(tmp_wav, sr=sr, mono=True)
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
        return y

    # fallback (may fail depending on backend)
    try:
        y, _ = librosa.load(mp4_path, sr=sr, mono=True)
        return y
    except Exception:
        return None


# --------------------------
# Binning
# --------------------------
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


# --------------------------
# Bin metrics (pred-level)
# --------------------------
def bin_metrics(df: pd.DataFrame, pred_col: str, threshold: float = 0.0) -> pd.DataFrame:
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


# --------------------------
# Prosody (per clip) + aggregate by bin
# --------------------------
def compute_prosody_features(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 500.0
) -> dict:
    if y is None or y.size == 0:
        return {}

    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if y_trim.size < int(0.2 * sr):
        y_trim = y

    duration_s = float(len(y_trim) / sr)

    rms = librosa.feature.rms(y=y_trim, frame_length=4 * hop_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    rms_mean = float(np.mean(rms_db))
    rms_std = float(np.std(rms_db))
    rms_p95 = float(np.percentile(rms_db, 95))
    rms_p05 = float(np.percentile(rms_db, 5))
    rms_range = float(rms_p95 - rms_p05)

    thr = np.median(rms) * 0.5
    pause_ratio = float(np.mean(rms < thr))

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_trim, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    voiced_mask = ~np.isnan(f0)
    voiced_ratio = float(np.mean(voiced_mask))

    if np.any(voiced_mask):
        f0_v = f0[voiced_mask].astype(np.float64)
        f0_mean = float(np.mean(f0_v))
        f0_median = float(np.median(f0_v))
        f0_std = float(np.std(f0_v))
        f0_min = float(np.min(f0_v))
        f0_max = float(np.max(f0_v))
        f0_range = float(f0_max - f0_min)

        t = np.arange(len(f0), dtype=np.float64)[voiced_mask]
        if t.size >= 2:
            A = np.vstack([t, np.ones_like(t)]).T
            slope, _ = np.linalg.lstsq(A, f0_v, rcond=None)[0]
            f0_slope = float(slope)
        else:
            f0_slope = float("nan")
    else:
        f0_mean = f0_median = f0_std = f0_range = f0_slope = float("nan")

    return {
        "duration_s": duration_s,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_range_p95_p05": rms_range,
        "pause_ratio": pause_ratio,
        "voiced_ratio": voiced_ratio,
        "f0_mean_hz": f0_mean,
        "f0_median_hz": f0_median,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "f0_slope": f0_slope,
    }


def add_prosody_per_row(df: pd.DataFrame, raw_dir: str, sr: int = 16000) -> pd.DataFrame:
    """
    Adds per-clip prosody features to df (one row per id).
    If file missing/decoding fails, keeps NaNs for those features.
    """
    df = df.copy()

    # build paths
    vids, clips, paths, exists = [], [], [], []
    for x in df["id"].astype(str).tolist():
        vid, clip = parse_id(x)
        vids.append(vid)
        clips.append(clip)
        p = None
        if vid is not None and clip is not None:
            p = video_path_from_id(raw_dir, vid, clip)
        paths.append(p)
        exists.append(bool(p) and os.path.exists(p))

    df["video_id"] = vids
    df["clip_id"] = clips
    df["video_path"] = paths
    df["video_exists"] = exists

    rows = []
    for i, r in df.iterrows():
        base = r.to_dict()
        if not r["video_exists"]:
            rows.append(base)  # no prosody
            continue

        y = load_audio_from_mp4(r["video_path"], sr=sr)
        if y is None:
            rows.append(base)
            continue

        feats = compute_prosody_features(y, sr=sr)
        rows.append({**base, **feats})

        if (i + 1) % 100 == 0:
            print(f"[prosody] processed {i+1}/{len(df)}", flush=True)

    return pd.DataFrame(rows)


def prosody_by_bin(df_with_prosody: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates prosody features by sentiment_class.
    Returns columns: bin, prosody_<feat>_mean, prosody_<feat>_std
    """
    prosody_cols = [
        "duration_s", "rms_mean", "rms_std", "rms_range_p95_p05",
        "pause_ratio", "voiced_ratio",
        "f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_range_hz", "f0_slope",
    ]
    prosody_cols = [c for c in prosody_cols if c in df_with_prosody.columns]

    rows = []
    for b, g in df_with_prosody.groupby("sentiment_class", dropna=False):
        rec = {"bin": str(b)}
        for c in prosody_cols:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            rec[f"prosody_{c}_mean"] = float(np.nanmean(x))
            rec[f"prosody_{c}_std"] = float(np.nanstd(x))
        rows.append(rec)

    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


# --------------------------
# Main
# --------------------------
def main(
    audio_json: str,
    vision_json: str,
    out_prefix: str,
    raw_dir: str,
    top_k: int = 200,
    sr: int = 16000,
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

    audio_bin.to_csv(f"{out_prefix}_bin_metrics_audio.csv", index=False, encoding="utf-8-sig")
    vision_bin.to_csv(f"{out_prefix}_bin_metrics_vision.csv", index=False, encoding="utf-8-sig")

    comb = audio_bin.merge(
        vision_bin,
        on="bin",
        suffixes=("_audio", "_vision"),
        how="outer"
    )

    # --------------------------
    # NEW: Prosody by bin (from RAW clips)
    # --------------------------
    print("[step] extracting prosody for all aligned clips (for bin-level aggregation)...")

    df_pros = add_prosody_per_row(df[["id", "sentiment_class"]].drop_duplicates().merge(
        df[["id"]], on="id", how="inner"
    ).merge(df[["id", "sentiment_class"]], on="id", how="left"), raw_dir=raw_dir, sr=sr)
    print('df_pros columns:', df_pros.columns)

    # Ensure bin column name matches comb merge key
    df_pros = df_pros.merge(df[["id", "sentiment_class"]].drop_duplicates(), on="id", how="left")
    print('df_pros columns:', df_pros.columns)
    pros_bin = prosody_by_bin(df_pros)
    print('pros_bin columns:', pros_bin.columns) 
   
    # Merge prosody stats into the side-by-side metrics
    comb = comb.merge(pros_bin, on="bin", how="left")
    print('comb columns:', comb.columns)  
    comb.to_csv(f"{out_prefix}_bin_metrics_side_by_side.csv", index=False, encoding="utf-8-sig")

    # --------------------------
    # (3) “Audio wins” examples
    # --------------------------
    audio_wins = df.sort_values("delta_audio_better", ascending=False).head(top_k)
    vision_wins = df.sort_values("delta_audio_better", ascending=True).head(top_k)

    keep_cols = [
        "id", "raw_text", "y_true", "sentiment_class",
        "yhat_audio", "yhat_vision",
        "abs_err_audio", "abs_err_vision",
        "delta_audio_better",
    ]
    audio_wins[keep_cols].to_csv(f"{out_prefix}_top_audio_wins.csv", index=False, encoding="utf-8-sig")
    vision_wins[keep_cols].to_csv(f"{out_prefix}_top_vision_wins.csv", index=False, encoding="utf-8-sig")

    # --------------------------
    # (5) Audio win rate by bin (+ avg abs error gap)
    # --------------------------
    df["audio_beats_vision"] = (df["delta_audio_better"] > 0).astype(int)
    df["abs_err_gap"] = (df["abs_err_vision"] - df["abs_err_audio"]).abs()

    win_rate = (
        df.groupby("sentiment_class")
          .agg(
              n=("id", "count"),
              audio_win_rate=("audio_beats_vision", "mean"),
              avg_abs_err_gap=("abs_err_gap", "mean"),
              avg_signed_err_gap=("delta_audio_better", "mean"),
          )
          .reset_index()
    )
    win_rate.to_csv(f"{out_prefix}_audio_win_rate_by_bin.csv", index=False, encoding="utf-8-sig")

    df.to_csv(f"{out_prefix}_aligned_audio_vs_vision.csv", index=False, encoding="utf-8-sig")

    print("[done] wrote:")
    print(f"  {out_prefix}_bin_metrics_audio.csv")
    print(f"  {out_prefix}_bin_metrics_vision.csv")
    print(f"  {out_prefix}_bin_metrics_side_by_side.csv  (NOW includes prosody_* columns)")
    print(f"  {out_prefix}_audio_win_rate_by_bin.csv")
    print(f"  {out_prefix}_top_audio_wins.csv")
    print(f"  {out_prefix}_top_vision_wins.csv")
    print(f"  {out_prefix}_aligned_audio_vs_vision.csv")


if __name__ == "__main__":
    # -------- EDIT THESE PATHS --------
    AUDIO_JSON = "/hpctmp/scratch/e0968015/mosei/FusionRuns/audio_guided_attn_run3/preds_test.json"
    VISION_JSON = "/hpctmp/scratch/e0968015/mosei/FusionRuns/vision_guided_attn_run3/preds_test.json"
    OUT_PREFIX = "/hpctmp/scratch/e0968015/mosei/FusionRuns/cluster_analysis_mosei_a35_run1/mosei_A35"

    # CHSIMS raw clips root:
    RAW_DIR = "/scratch/e0968015/mosei/Raw"

    os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)
    main(AUDIO_JSON, VISION_JSON, OUT_PREFIX, raw_dir=RAW_DIR, top_k=300, sr=16000)
