#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
from collections import defaultdict

import cv2
import mediapipe as mp
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
    return str(Path(raw_dir) / videoid / f"{clipid}.mp4")


def try_extract_wav_with_ffmpeg(video_path: str, wav_path: str, sr: int = 16000) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def load_audio_from_mp4(mp4_path: str, sr: int = 16000):
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
# MediaPipe visual helpers
# --------------------------
def make_facelandmarker(model_path: str, num_faces: int = 1):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=num_faces,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(opts)


def compute_visual_features_mediapipe(
    video_path: str,
    landmarker,
    target_fps: float = 5.0,
    max_frames: int = 300,
) -> dict:
    """
    Returns aggregated per-clip facial features:
      - face_success_rate
      - visual_bs_* mean/std/max
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"face_success_rate": float("nan"), "face_n_frames": 0, "face_n_success": 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    stride = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1

    frame_idx = 0
    used = 0
    success = 0
    bs_scores = defaultdict(list)

    while used < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int(frame_idx * 1000.0 / fps)

        try:
            res = landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            used += 1
            frame_idx += 1
            continue

        used += 1

        if res.face_blendshapes and len(res.face_blendshapes) > 0:
            success += 1
            for cat in res.face_blendshapes[0]:
                bs_scores[cat.category_name].append(float(cat.score))

        frame_idx += 1

    cap.release()

    out = {
        "face_n_frames": int(used),
        "face_n_success": int(success),
        "face_success_rate": float(success / used) if used > 0 else float("nan"),
    }

    for name, vals in bs_scores.items():
        v = np.asarray(vals, dtype=np.float32)
        out[f"visual_bs_{name}_mean"] = float(np.mean(v))
        out[f"visual_bs_{name}_std"] = float(np.std(v))
        out[f"visual_bs_{name}_max"] = float(np.max(v))

    return out


# --------------------------
# Binning
# --------------------------
def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    5-class label grouping based on y_true values (rounded to 1dp):
      - negative:        [-1.0, -0.8]
      - weakly negative: [-0.6, -0.4, -0.2]
      - neutral:         [ 0.0]
      - weakly positive: [ 0.2,  0.4,  0.6]
      - positive:        [ 0.8,  1.0]
    """
    df = df.copy()
    y = df["y_true"].round(1)

    mapping = {
        -1.0: "negative",
        -0.8: "negative",
        -0.6: "weakly_negative",
        -0.4: "weakly_negative",
        -0.2: "weakly_negative",
         0.0: "neutral",
         0.2: "weakly_positive",
         0.4: "weakly_positive",
         0.6: "weakly_positive",
         0.8: "positive",
         1.0: "positive",
    }

    df["sentiment_class"] = y.map(mapping)
    unknown = df["sentiment_class"].isna()
    if unknown.any():
        df.loc[unknown, "sentiment_class"] = "other"
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
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_p95 = float(np.percentile(rms, 95))
    rms_p05 = float(np.percentile(rms, 5))
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
            rows.append(base)
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

def add_visual_per_row(
    df: pd.DataFrame,
    raw_dir: str,
    face_model: str,
    face_fps: float = 5.0,
    face_max_frames: int = 300,
) -> pd.DataFrame:
    """
    Adds per-clip visual features to df (one row per id).
    If file missing/decoding fails, keeps NaNs for those features.
    Also stores visual extraction errors in a visual_error column.
    """
    df = df.copy()

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
    with make_facelandmarker(face_model, num_faces=1) as landmarker:
        for i, r in df.iterrows():
            base = r.to_dict()
            base["visual_error"] = ""

            if not r["video_exists"]:
                base["visual_error"] = "missing_video_file"
                rows.append(base)
                continue

            try:
                feats = compute_visual_features_mediapipe(
                    r["video_path"],
                    landmarker,
                    target_fps=face_fps,
                    max_frames=face_max_frames,
                )

                # optionally flag rows where MediaPipe ran but found no face
                if feats.get("face_n_success", 0) == 0:
                    base["visual_error"] = "no_face_detected"

                rows.append({**base, **feats})

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                base["visual_error"] = err_msg
                print(f"[visual] failed for {r['video_path']}: {err_msg}", flush=True)
                rows.append(base)

            if (i + 1) % 100 == 0:
                print(f"[visual] processed {i+1}/{len(df)}", flush=True)

    return pd.DataFrame(rows)

def visual_by_bin(df_with_visual: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates visual features by sentiment_class.
    Returns columns: bin, visual_*_mean, visual_*_std
    """
    visual_cols = [c for c in df_with_visual.columns if c.startswith("visual_bs_")]
    extra_cols = [c for c in ["face_n_frames", "face_n_success", "face_success_rate"] if c in df_with_visual.columns]
    visual_cols = extra_cols + visual_cols

    rows = []
    for b, g in df_with_visual.groupby("sentiment_class", dropna=False):
        rec = {"bin": str(b)}
        for c in visual_cols:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            rec[f"{c}_mean"] = float(np.nanmean(x))
            rec[f"{c}_std"] = float(np.nanstd(x))
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
    face_model: str,
    top_k: int = 200,
    sr: int = 16000,
    face_fps: float = 5.0,
    face_max_frames: int = 300,
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
    print(comb)

    # --------------------------
    # Prosody by bin (from RAW clips)
    # --------------------------
    print("[step] extracting prosody for all aligned clips (for bin-level aggregation)...")
    base_clip_df = df[["id", "sentiment_class"]].drop_duplicates().copy()
    df_pros = add_prosody_per_row(base_clip_df, raw_dir=raw_dir, sr=sr)
    pros_bin = prosody_by_bin(df_pros)

    # --------------------------
    # Visual by bin (from RAW clips)
    # --------------------------
    print("[step] extracting visual features for all aligned clips (for bin-level aggregation)...")
    df_vis = add_visual_per_row(
        base_clip_df,
        raw_dir=raw_dir,
        face_model=face_model,
        face_fps=face_fps,
        face_max_frames=face_max_frames,
    )
    vis_bin = visual_by_bin(df_vis)

    # Merge prosody stats + visual stats into side-by-side metrics
    comb = comb.merge(pros_bin, on="bin", how="left")
    comb = comb.merge(vis_bin, on="bin", how="left")
    comb.to_csv(f"{out_prefix}_bin_metrics_side_by_side.csv", index=False, encoding="utf-8-sig")

    # Optional: save row-level extracted features too
    df_pros.to_csv(f"{out_prefix}_prosody_per_clip.csv", index=False, encoding="utf-8-sig")
    df_vis.to_csv(f"{out_prefix}_visual_per_clip.csv", index=False, encoding="utf-8-sig")

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
    print(f"  {out_prefix}_bin_metrics_side_by_side.csv  (NOW includes prosody_* and visual_* columns)")
    print(f"  {out_prefix}_prosody_per_clip.csv")
    print(f"  {out_prefix}_visual_per_clip.csv")
    print(f"  {out_prefix}_audio_win_rate_by_bin.csv")
    print(f"  {out_prefix}_top_audio_wins.csv")
    print(f"  {out_prefix}_top_vision_wins.csv")
    print(f"  {out_prefix}_aligned_audio_vs_vision.csv")


if __name__ == "__main__":
    # -------- EDIT THESE PATHS --------
    AUDIO_JSON = "/hpctmp/scratch/e0968015/chsims/FusionRuns/audio_guided_attn_run3/preds_test.json"
    VISION_JSON = "/hpctmp/scratch/e0968015/chsims/FusionRuns/vision_guided_attn_run3/preds_test.json"
    OUT_PREFIX = "/hpctmp/scratch/e0968015/chsims/FusionRuns/posthoc_audio_vs_vision_run3/chsims_A35"

    # CHSIMS raw clips root:
    RAW_DIR = "/hpctmp/scratch/e0968015/chsims/Raw"

    # MediaPipe face model:
    FACE_MODEL = "/hpctmp/scratch/e0968015/models/face_landmarker.task"

    os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)
    main(
        AUDIO_JSON,
        VISION_JSON,
        OUT_PREFIX,
        raw_dir=RAW_DIR,
        face_model=FACE_MODEL,
        top_k=300,
        sr=16000,
        face_fps=5.0,
        face_max_frames=300,
    )
