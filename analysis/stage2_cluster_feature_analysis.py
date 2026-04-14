#!/usr/bin/env python3
"""
Stage-2 cluster posthoc analysis: audio + visual + text features per clip, aggregated by cluster2.

Input: stage2_clustered_union.csv (from your stage2 clustering script)
Required columns:
  - id (formatted like videoid$_$clipid)
  - raw_text
  - y_true
  - cluster2
Optional:
  - yhat__* columns (kept for reference)

Raw clips expected at:
  --raw-dir/<video_id>/<clip_id>.mp4

Visual features:
  MediaPipe FaceLandmarker in VIDEO mode (you decode frames using OpenCV).
  Needs --face-model pointing to face_landmarker.task

Outputs (in --out-dir):
  - stage2_clip_features.csv
  - stage2_cluster_profiles.csv
  - stage2_cluster_top_tokens.csv
  - stage2_failures.csv
"""

import os
import re
import csv
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# pandas makes grouping / output easier (your earlier pipelines already use it)
import pandas as pd

# Audio
import librosa
import subprocess

# Video decode
import cv2

# MediaPipe
import mediapipe as mp


# -----------------------
# Helpers: ID -> filepath
# -----------------------
def parse_id(id_str: str):
    s = str(id_str)
    if "$_$" not in s:
        return None, None
    vid, clip = s.split("$_$", 1)
    return vid, clip

def clip_path(raw_dir: str, video_id: str, clip_id: str) -> str:
    return str(Path(raw_dir) / video_id / f"{clip_id}.mp4")


# -----------------------
# Audio loading (ffmpeg)
# -----------------------
def try_extract_wav_with_ffmpeg(video_path: str, wav_path: str, sr: int = 16000) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def load_audio_from_mp4(mp4_path: str, sr: int = 16000) -> np.ndarray:
    tmp_wav = str(Path(mp4_path).with_suffix(".tmp16k.wav"))
    if try_extract_wav_with_ffmpeg(mp4_path, tmp_wav, sr=sr) and os.path.exists(tmp_wav):
        y, _ = librosa.load(tmp_wav, sr=sr, mono=True)
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
        return y
    # fallback
    y, _ = librosa.load(mp4_path, sr=sr, mono=True)
    return y


# -----------------------
# Audio features (librosa)
# -----------------------
def compute_audio_features(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 500.0,
    n_mfcc: int = 13,
) -> dict:
    if y is None or y.size == 0:
        return {}

    # trim silence
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if y_trim.size < int(0.2 * sr):
        y_trim = y

    duration_s = float(len(y_trim) / sr)

    # RMS energy
    rms = librosa.feature.rms(y=y_trim, frame_length=4 * hop_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    thr = np.median(rms) * 0.5
    pause_ratio = float(np.mean(rms < thr))

    # Pitch via pYIN
    f0, voiced_flag, voiced_prob = librosa.pyin(y_trim, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    voiced_mask = ~np.isnan(f0)
    voiced_ratio = float(np.mean(voiced_mask))

    if np.any(voiced_mask):
        f0_v = f0[voiced_mask].astype(np.float64)
        f0_mean = float(np.mean(f0_v))
        f0_std = float(np.std(f0_v))
        f0_min = float(np.min(f0_v))
        f0_max = float(np.max(f0_v))
        f0_range = float(f0_max - f0_min)
    else:
        f0_mean = f0_std = f0_min = f0_max = f0_range = float("nan")

    # MFCC summary (mean/std across time; then summarize across coeffs)
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)  # (n_mfcc, T)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # compress mfcc vectors into scalars (still informative for clustering)
    mfcc_mean_mean = float(np.mean(mfcc_mean))
    mfcc_mean_std  = float(np.std(mfcc_mean))
    mfcc_std_mean  = float(np.mean(mfcc_std))
    mfcc_std_std   = float(np.std(mfcc_std))

    return {
        "duration_s": duration_s,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "pause_ratio": pause_ratio,
        "voiced_ratio": voiced_ratio,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "mfcc_mean_mean": mfcc_mean_mean,
        "mfcc_mean_std": mfcc_mean_std,
        "mfcc_std_mean": mfcc_std_mean,
        "mfcc_std_std": mfcc_std_std,
    }


# -----------------------
# Text features
# -----------------------
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# very small fallback zh stoplist (you can expand or pass a file if you want later)
_ZH_STOP = set(list("的了是在我你他她它们我们你们他们她们这那就都也很还与及和而"))
_PUNCTS = ["?", "？", "!", "！", ".", "。", ",", "，", "…", ":", "：", ";", "；"]

def load_english_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        # fallback minimal
        return {
            "the","a","an","and","or","but","if","then","else","to","of","in","on","at","for","from",
            "is","are","was","were","be","been","being","i","me","my","we","our","you","your","he","she",
            "it","they","them","this","that","these","those"
        }

_EN_STOP = load_english_stopwords()

def detect_lang(text: str) -> str:
    # crude: if contains CJK -> zh, else en
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"

def tokenize_en(text: str):
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    toks = [t for t in toks if t not in _EN_STOP and len(t) > 1]
    return toks

def tokenize_zh(text: str):
    # no jieba dependency: use per-character tokens but drop common function chars
    toks = [ch for ch in text if ("\u4e00" <= ch <= "\u9fff") and (ch not in _ZH_STOP)]
    return toks

def compute_text_features(raw_text: str) -> dict:
    s = "" if raw_text is None else str(raw_text)
    lang = detect_lang(s)

    # lengths
    text_len_chars = len(s)

    # tokens + top tokens to be aggregated later
    if lang == "en":
        toks = tokenize_en(s)
    else:
        toks = tokenize_zh(s)

    return {
        "lang_guess": lang,
        "text_len_chars": text_len_chars,
        "token_count": int(len(toks)),
    }, toks


# -----------------------
# Visual features (MediaPipe)
# -----------------------
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
      - blendshape_* mean/std/max (averaged across detected frames)
    """
    print(f"[video] {video_path}")
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

    # collect blendshape scores per frame (dict name -> list)
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
        print('Monotonically increasing timestamp:', ts_ms)

        try:
            res = landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as e:
            # treat as failed frame
            used += 1
            frame_idx += 1
            print(f"[mediapipe error] video={video_path} frame_idx={frame_idx} ts_ms={ts_ms} error={e}", flush=True)
            continue

        used += 1
        has_landmarks = bool(getattr(res, "face_landmarks", None)) and len(res.face_landmarks) > 0
        has_blend = bool(getattr(res, "face_blendshapes", None)) and len(res.face_blendshapes) > 0
        print('Has landmarks:', has_landmarks)
        print('Has blendshapes:', has_blend)

        if res.face_blendshapes and len(res.face_blendshapes) > 0:
            success += 1
            # take first face
            for cat in res.face_blendshapes[0]:
                bs_scores[cat.category_name].append(float(cat.score))
        print('bs_score:', dict(bs_scores))
        frame_idx += 1

    cap.release()

    out = {
        "face_n_frames": int(used),
        "face_n_success": int(success),
        "face_success_rate": float(success / used) if used > 0 else float("nan"),
    }

    # aggregate blendshapes
    for name, vals in bs_scores.items():
        v = np.asarray(vals, dtype=np.float32)
        out[f"bs_{name}_mean"] = float(np.mean(v))
        out[f"bs_{name}_std"]  = float(np.std(v))
        out[f"bs_{name}_max"]  = float(np.max(v))
    print(out)

    return out


# -----------------------
# Main pipeline
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=str, required=True, help="stage2_clustered_union.csv")
    ap.add_argument("--raw-dir", type=str, required=True, help="Raw clip root dir: <video_id>/<clip_id>.mp4")
    ap.add_argument("--out-dir", type=str, required=True, help="Output dir")
    ap.add_argument("--sr", type=int, default=16000, help="Audio sample rate")
    ap.add_argument("--face-model", type=str, required=True, help="Path to face_landmarker.task")
    ap.add_argument("--face-fps", type=float, default=5.0, help="FPS to sample for MediaPipe (e.g. 5)")
    ap.add_argument("--face-max-frames", type=int, default=300, help="Max sampled frames per clip")
    ap.add_argument("--limit", type=int, default=0, help="Debug: limit number of clips (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    for c in ["id", "raw_text", "y_true", "cluster2"]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column in input CSV: {c}")

    # keep yhat cols for reference (optional)
    yhat_cols = [c for c in df.columns if c.startswith("yhat__")]

    if args.limit and args.limit > 0:
        df = df.head(int(args.limit)).copy()

    # resolve paths
    vids, clips, paths, exists = [], [], [], []
    for x in df["id"].astype(str).tolist():
        vid, clip = parse_id(x)
        vids.append(vid)
        clips.append(clip)
        p = clip_path(args.raw_dir, vid, clip) if vid and clip else ""
        paths.append(p)
        exists.append(bool(p) and os.path.exists(p))

    df["video_id"] = vids
    df["clip_id"] = clips
    df["video_path"] = paths
    df["exists"] = exists

    print(f"[info] rows={len(df)}")
    print(f"[info] missing files={(~df['exists']).sum()}")

    # init FaceLandmarker once
    print("[info] loading MediaPipe FaceLandmarker...")
    landmarker = make_facelandmarker(args.face_model, num_faces=1)

    rows = []
    fails = []

    try:
        for i, r in df.iterrows():
            base = {
                "id": r["id"],
                "video_id": r["video_id"],
                "clip_id": r["clip_id"],
                "video_path": r["video_path"],
                "cluster2": int(r["cluster2"]),
                "y_true": float(r["y_true"]),
                "raw_text": "" if pd.isna(r["raw_text"]) else str(r["raw_text"]),
            }
            for c in yhat_cols:
                base[c] = r[c]

            if not r["exists"]:
                fails.append({**base, "error": "missing_video_file"})
                continue

            # --- text ---
            t_feats, toks = compute_text_features(base["raw_text"])

            # --- audio ---
            try:
                y = load_audio_from_mp4(base["video_path"], sr=args.sr)
                a_feats = compute_audio_features(y, sr=args.sr)
            except Exception as e:
                a_feats = {}
                fails.append({**base, "error": f"audio_fail: {e}"})

            # --- visual ---
            try:
                with make_facelandmarker(args.face_model, num_faces=1) as landmarker:
                    v_feats = compute_visual_features_mediapipe(
                        base["video_path"], landmarker,
                        target_fps=args.face_fps,
                        max_frames=args.face_max_frames,
                    )
            except Exception as e:
                v_feats = {}
                fails.append({**base, "error": f"vision_fail: {e}"})

            # store tokens for later aggregation
            # (we keep them in a single string column for now)
            tok_str = " ".join(toks) if toks else ""

            rows.append({
                **base,
                **t_feats,
                **a_feats,
                **v_feats,
                "tokens_clean": tok_str,
            })

            if (i + 1) % 50 == 0:
                print(f"[progress] {i+1}/{len(df)} processed", flush=True)
            
    finally: 
        print("[info] finished processing loop")

    feat_df = pd.DataFrame(rows)
    fail_df = pd.DataFrame(fails)

    out_feat = out_dir / "stage2_clip_features.csv"
    out_fail = out_dir / "stage2_failures.csv"
    feat_df.to_csv(out_feat, index=False, encoding="utf-8-sig")
    fail_df.to_csv(out_fail, index=False, encoding="utf-8-sig")
    print("[done] wrote:", out_feat)
    print("[done] wrote:", out_fail)

    if feat_df.empty:
        print("[warn] No successful rows to summarise.")
        return

    # -------------------------
    # Cluster profiles (mean/std)
    # -------------------------
    # numeric cols only (exclude huge token strings, ids)
    exclude = {"id","video_id","clip_id","video_path","raw_text","tokens_clean","lang_guess"}
    num_cols = [c for c in feat_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(feat_df[c])]

    prof_rows = []
    for c2, g in feat_df.groupby("cluster2"):
        rec = {"cluster2": int(c2), "n": int(len(g))}
        for c in num_cols:
            x = pd.to_numeric(g[c], errors="coerce")
            rec[f"{c}_mean"] = float(np.nanmean(x.to_numpy(dtype=float)))
            rec[f"{c}_std"]  = float(np.nanstd(x.to_numpy(dtype=float)))
        prof_rows.append(rec)

    prof_df = pd.DataFrame(prof_rows).sort_values("cluster2")
    out_prof = out_dir / "stage2_cluster_profiles.csv"
    prof_df.to_csv(out_prof, index=False, encoding="utf-8-sig")
    print("[done] wrote:", out_prof)

    # -------------------------
    # Top tokens per cluster
    # -------------------------
    tok_rows = []
    for c2, g in feat_df.groupby("cluster2"):
        counter = Counter()
        for s in g["tokens_clean"].fillna("").astype(str).tolist():
            if not s:
                continue
            counter.update(s.split())
        # top 30
        for tok, cnt in counter.most_common(30):
            tok_rows.append({"cluster2": int(c2), "token": tok, "count": int(cnt)})

    tok_df = pd.DataFrame(tok_rows).sort_values(["cluster2","count"], ascending=[True, False])
    out_tok = out_dir / "stage2_cluster_top_tokens.csv"
    tok_df.to_csv(out_tok, index=False, encoding="utf-8-sig")
    print("[done] wrote:", out_tok)

    print("\n[summary]")
    print("clusters:", sorted(feat_df["cluster2"].unique().tolist()))
    print("rows per cluster:")
    print(feat_df.groupby("cluster2").size().to_string())


if __name__ == "__main__":
    main()
