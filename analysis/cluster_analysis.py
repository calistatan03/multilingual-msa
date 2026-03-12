#!/usr/bin/env python3
"""
Post-hoc cluster clip analysis (NO OpenFace):
1) Prediction-pattern engineered features (from yhat__<method> columns)
2) Audio features (librosa + ffmpeg wav extraction)
3) Text engineered features (lightweight, EN + ZH)

Inputs:
- One or more per-cluster CSVs produced by your clustering script, OR a directory containing them.
  Expected columns include:
    id, raw_text, y_true, cluster, yhat__Audio-Guided Attention, yhat__Vision-Guided Attention, ...

- Raw clip directory root (dataset Raw):
    CHSIMS example: /scratch/e0968015/chsims/Raw/<video_id>/<clip_id>.mp4
    MOSEI example:  /scratch/e0968015/mosei/Raw/<video_id>/<clip_id>.mp4

Outputs (in --out-dir):
- utterances_enriched.csv   (row per clip with all engineered features)
- cluster_profile.csv       (cluster-level means/stds for selected features)
- missing_files.csv         (clips not found)

Usage examples:
  python cluster_clip_analysis.py \
    --inputs /scratch/e0968015/chsims/FusionRuns/cluster_analysis/chsims_A35/abs_cluster0_clips_with_predictions.csv \
    --raw-dir /scratch/e0968015/chsims/Raw \
    --out-dir /scratch/e0968015/chsims/FusionRuns/cluster_analysis/chsims_A35/posthoc

  python cluster_clip_analysis.py \
    --inputs /scratch/e0968015/chsims/FusionRuns/cluster_analysis/chsims_A35 \
    --raw-dir /scratch/e0968015/chsims/Raw \
    --out-dir /scratch/e0968015/chsims/FusionRuns/cluster_analysis/chsims_A35/posthoc \
    --glob "*clips_with_predictions.csv"

Notes:
- Requires ffmpeg available in PATH for reliable mp4->wav extraction.
- Keeps runtime simple (sequential). If you want multiprocessing later, we can add it safely.
"""

import os
import re
import glob
import math
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# --------------------------
# ID parsing / path building
# --------------------------
def split_video_clip_id(id_str: str) -> Tuple[str, Optional[str]]:
    s = str(id_str)
    if "$_$" not in s:
        return s, None
    vid, clip = s.split("$_$", 1)
    return vid, clip


def build_video_path(raw_dir: str, video_id: str, clip_id: Optional[str]) -> Optional[str]:
    if not video_id or clip_id is None:
        return None
    return str(Path(raw_dir) / video_id / f"{clip_id}.mp4")


# --------------------------
# Prediction-pattern features
# --------------------------
def get_method_cols(df: pd.DataFrame) -> List[str]:
    # yhat columns are named like yhat__Kernel Fusion
    return [c for c in df.columns if c.startswith("yhat__")]


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def add_prediction_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric y_true
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

    yhat_cols = get_method_cols(df)
    if not yhat_cols:
        raise ValueError("No yhat__<method> columns found. Did you pass the correct per-cluster CSVs?")

    # Ensure numeric yhat columns
    for c in yhat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Per-method errors
    for c in yhat_cols:
        m = c.replace("yhat__", "")
        df[f"abs_err__{m}"] = (df[c] - df["y_true"]).abs()
        df[f"signed_err__{m}"] = (df[c] - df["y_true"])

    # Cross-model prediction disagreement on yhat (not error)
    yhat_mat = df[yhat_cols].to_numpy(dtype=np.float64)
    df["pred_mean"] = np.nanmean(yhat_mat, axis=1)
    df["pred_std"] = np.nanstd(yhat_mat, axis=1)
    df["pred_min"] = np.nanmin(yhat_mat, axis=1)
    df["pred_max"] = np.nanmax(yhat_mat, axis=1)
    df["pred_range"] = df["pred_max"] - df["pred_min"]

    # Identify best/worst method per utterance (by abs error)
    abs_err_cols = [c for c in df.columns if c.startswith("abs_err__")]
    abs_err_mat = df[abs_err_cols].to_numpy(dtype=np.float64)

    best_idx = np.nanargmin(abs_err_mat, axis=1)
    worst_idx = np.nanargmax(abs_err_mat, axis=1)

    best_methods = [abs_err_cols[i].replace("abs_err__", "") for i in best_idx]
    worst_methods = [abs_err_cols[i].replace("abs_err__", "") for i in worst_idx]
    df["best_method"] = best_methods
    df["worst_method"] = worst_methods

    # Optional: audio-vs-vision delta if both present
    # (positive => audio model has lower abs error)
    audio_key = "Audio-Guided Attention"
    vision_key = "Vision-Guided Attention"
    a_col = f"abs_err__{audio_key}"
    v_col = f"abs_err__{vision_key}"
    # if a_col in df.columns and v_col in df.columns:
        # df["delta_audio_better"] = df[v_col] - df[a_col]
        # df["audio_beats_vision"] = (df["delta_audio_better"] > 0).astype(int)
        # df["abs_err_gap_audio_vision"] = (df[v_col] - df[a_col]).abs()

    return df


# --------------------------
# Audio features (librosa)
# --------------------------
def ffmpeg_to_wav(mp4_path: str, wav_path: str, sr: int = 16000) -> bool:
    cmd = ["ffmpeg", "-y", "-i", mp4_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def load_audio_from_mp4(mp4_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    tmp_wav = str(Path(mp4_path).with_suffix(f".tmp{sr}.wav"))
    if ffmpeg_to_wav(mp4_path, tmp_wav, sr=sr) and os.path.exists(tmp_wav):
        y, _ = librosa.load(tmp_wav, sr=sr, mono=True)
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
        return y

    # fallback (may fail if backend can't decode)
    try:
        y, _ = librosa.load(mp4_path, sr=sr, mono=True)
        return y
    except Exception:
        return None


def compute_audio_features(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 500.0,
    n_mfcc: int = 13
) -> Dict[str, float]:
    if y is None or y.size == 0:
        return {}

    # Trim silences a bit (keeps stability)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if y_trim.size < int(0.2 * sr):
        y_trim = y

    duration_s = float(len(y_trim) / sr)

    # RMS
    rms = librosa.feature.rms(y=y_trim, frame_length=4 * hop_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    thr = np.median(rms) * 0.5
    pause_ratio = float(np.mean(rms < thr))

    # Pitch with pyin
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_trim, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
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

    # MFCC summary
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)  # (n_mfcc, T)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feats = {
        "duration_s": duration_s,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "pause_ratio": pause_ratio,
        "voiced_ratio": voiced_ratio,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
    }
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
        feats[f"mfcc{i+1}_std"] = float(mfcc_std[i])

    return feats


def add_audio_features(df: pd.DataFrame, raw_dir: str, sr: int = 16000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Parse ids to paths
    vids, clips, paths, exists = [], [], [], []
    for x in df["id"].astype(str).tolist():
        vid, clip = split_video_clip_id(x)
        p = build_video_path(raw_dir, vid, clip)
        vids.append(vid)
        clips.append(clip)
        paths.append(p)
        exists.append(bool(p) and os.path.exists(p))

    df["video_id"] = vids
    df["clip_id"] = clips
    df["video_path"] = paths
    df["video_exists"] = exists

    missing_df = df.loc[~df["video_exists"], ["id", "video_id", "clip_id", "video_path"]].copy()

    rows = []
    for i, r in df.iterrows():
        if not r["video_exists"]:
            rows.append({**r.to_dict(), "audio_error": "missing_video"})
            continue

        try:
            y = load_audio_from_mp4(r["video_path"], sr=sr)
            if y is None:
                rows.append({**r.to_dict(), "audio_error": "audio_decode_failed"})
                continue
            feats = compute_audio_features(y, sr=sr)
            rows.append({**r.to_dict(), **feats})
        except Exception as e:
            rows.append({**r.to_dict(), "audio_error": str(e)})

        if (i + 1) % 50 == 0:
            print(f"[audio] processed {i+1}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    return out, missing_df


# --------------------------
# Text engineered features
# --------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# very lightweight lexicons (you can expand later)
ZH_NEG = ["不", "没", "無", "无", "别", "不是", "不会", "不能", "讨厌", "烦", "郁闷", "痛苦", "糟", "差"]
ZH_POS = ["好", "喜欢", "开心", "高兴", "棒", "不错", "满意", "幸福", "可爱", "赞", "细腻", "红润", "光泽"]

EN_NEG = ["not", "no", "never", "none", "nothing", "bad", "terrible", "awful", "hate", "sad", "angry", "upset"]
EN_POS = ["good", "great", "love", "happy", "amazing", "awesome", "nice", "excellent", "wonderful", "like"]


def detect_lang(text: str) -> str:
    if text is None:
        return "unknown"
    t = str(text)
    return "zh" if _CJK_RE.search(t) else "en"


def count_substrings(text: str, words: List[str]) -> int:
    t = str(text)
    return int(sum(t.count(w) for w in words))

import re
from collections import Counter

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")



# Basic stopwords (expand as needed)
EN_STOP = set(stopwords.words('english'))

# Very simple Chinese stopwords/particles (keep small + safe)
ZH_STOP = {
    "的","了","呢","啊","嘛","吧","呀","哦","哎","唉","哈","呵","呃",
    "我","你","他","她","它","我们","你们","他们","她们",
    "这","那","这些","那些","一个","一些","这个","那个",
    "在","是","有","和","与","及","也","都","就","才","还","又","很","更",
    "不","没","没有",  # you can remove these if you want to keep negations
}

PUNCTS = ["!", "?", "…", ".", ",", "。", "，", "！", "？", "；", "：", "、", "（", "）", "(", ")", "“", "”", "\"", "'"]


def detect_lang(text: str) -> str:
    if text is None:
        return "unknown"
    t = str(text)
    return "zh" if _CJK_RE.search(t) else "en"


def tokenize_en(text: str) -> list:
    toks = [w.lower() for w in _WORD_RE.findall(str(text).lower())]
    toks = [t for t in toks if t not in EN_STOP and len(t) >= 2]
    return toks


def tokenize_zh_chars(text: str) -> list:
    """
    Super-light baseline: treat each Chinese character as a token.
    Better segmentation would use jieba, but this works without extra deps.
    """
    chars = [ch for ch in str(text) if _CJK_RE.match(ch)]
    chars = [c for c in chars if c not in ZH_STOP]
    return chars


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    texts = df["raw_text"].fillna("").astype(str)

    df["lang_guess"] = texts.map(detect_lang)
    df["text_len_chars"] = texts.map(len)

    # crude token counts
    df["text_len_tokens_en"] = texts.map(lambda s: len(_WORD_RE.findall(s)))
    df["text_len_tokens_zh_chars"] = texts.map(lambda s: sum(1 for ch in s if _CJK_RE.match(ch)))

    # punctuation counts
    for p in PUNCTS:
        df[f"cnt_{ord(p)}"] = texts.map(lambda s, pp=p: s.count(pp))

    df["has_question"] = texts.map(lambda s: int(("?" in s) or ("？" in s)))
    df["has_exclaim"]  = texts.map(lambda s: int(("!" in s) or ("！" in s)))

    return df

def compute_top_tokens(df: pd.DataFrame, top_n: int = 50) -> dict:
    """
    Returns:
      {
        "en": [(token, count), ...],
        "zh": [(token, count), ...]
      }
    """
    en_counter = Counter()
    zh_counter = Counter()

    for txt in df["raw_text"].fillna("").astype(str).tolist():
        lang = detect_lang(txt)
        if lang == "zh":
            zh_counter.update(tokenize_zh_chars(txt))
        else:
            en_counter.update(tokenize_en(txt))

    return {
        "en": en_counter.most_common(top_n),
        "zh": zh_counter.most_common(top_n),
    }


def write_top_tokens_csv(top_tokens: dict, out_path: str):
    rows = []
    for lang, pairs in top_tokens.items():
        for tok, cnt in pairs:
            rows.append({"lang": lang, "token": tok, "count": int(cnt)})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def compute_top_tokens_by_cluster(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Returns a long table: cluster, lang, token, count
    """
    rows = []
    for cl, g in df.groupby("cluster", dropna=False):
        toks = compute_top_tokens(g, top_n=top_n)
        for lang, pairs in toks.items():
            for tok, cnt in pairs:
                rows.append({"cluster": cl, "lang": lang, "token": tok, "count": int(cnt)})
    return pd.DataFrame(rows).sort_values(["cluster", "lang", "count"], ascending=[True, True, False])
# --------------------------
# Cluster profiling
# --------------------------
def make_cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster-level aggregation for a compact, report-friendly table.
    """
    df = df.copy()

    # choose some useful columns if they exist
    cols = []
    for c in [
        "y_true",
        "pred_std", "pred_range",
        "voiced_ratio", "pause_ratio",
        "f0_mean_hz", "f0_std_hz", "f0_range_hz",
        "rms_mean", "rms_std",
        "text_len_chars", "text_len_words_en",
        "zh_neg_lex_count", "zh_pos_lex_count", "en_neg_lex_count", "en_pos_lex_count",
        "contrast_count",
    ]:
        if c in df.columns:
            cols.append(c)

    # include audio-vs-vision columns if present
    for c in ["audio_beats_vision", "abs_err_gap_audio_vision", "delta_audio_better"]:
        if c in df.columns:
            cols.append(c)

    if "cluster" not in df.columns:
        raise ValueError("Expected 'cluster' column in input.")

    rows = []
    for cl, g in df.groupby("cluster", dropna=False):
        rec = {"cluster": cl, "n": int(len(g))}
        for c in cols:
            x = pd.to_numeric(g[c], errors="coerce")
            rec[f"{c}_mean"] = float(np.nanmean(x))
            rec[f"{c}_std"] = float(np.nanstd(x))
        # common best/worst method if available
        if "best_method" in g.columns:
            rec["most_common_best_method"] = g["best_method"].value_counts().index[0]
        if "worst_method" in g.columns:
            rec["most_common_worst_method"] = g["worst_method"].value_counts().index[0]
        rows.append(rec)

    return pd.DataFrame(rows).sort_values("cluster")


# --------------------------
# IO: discover inputs
# --------------------------
def resolve_input_files(inputs: List[str], pattern: str) -> List[str]:
    files = []
    for x in inputs:
        p = Path(x)
        if p.is_file():
            files.append(str(p))
        elif p.is_dir():
            files.extend(sorted([str(pp) for pp in p.glob(pattern)]))
        else:
            # allow glob strings
            files.extend(sorted(glob.glob(x)))
    # de-dup while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV files or directories containing per-cluster CSVs")
    ap.add_argument("--glob", type=str, default="*clips_with_predictions.csv", help="Pattern for directory inputs")
    ap.add_argument("--raw-dir", type=str, required=True, help="Raw clip root dir (contains <video_id>/<clip_id>.mp4)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--sr", type=int, default=16000, help="Audio sampling rate for librosa")
    args = ap.parse_args()

    in_files = resolve_input_files(args.inputs, args.glob)
    if not in_files:
        raise SystemExit("No input CSV files found. Check --inputs and --glob.")

    print(f"[scan] Found {len(in_files)} input CSVs")
    for f in in_files[:20]:
        print("  -", f)
    if len(in_files) > 20:
        print("  ...")

    # Load and combine
    dfs = []
    for f in in_files:
        df = pd.read_csv(f)
        df["source_csv"] = os.path.basename(f)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Prediction features
    print("[step] prediction-pattern features...")
    df_all = add_prediction_pattern_features(df_all)

    # Text features
    print("[step] text engineered features...")
    df_all = add_text_features(df_all)

    # Top tokens overall + per cluster
    top_overall = compute_top_tokens(df_all, top_n=80)
    top_overall_csv = os.path.join(args.out_dir, "top_tokens_overall.csv")
    write_top_tokens_csv(top_overall, top_overall_csv)

    top_by_cluster = compute_top_tokens_by_cluster(df_all, top_n=40)
    top_by_cluster_csv = os.path.join(args.out_dir, "top_tokens_by_cluster.csv")
    top_by_cluster.to_csv(top_by_cluster_csv, index=False, encoding="utf-8-sig")

    # Audio features
    print("[step] audio features (librosa)...")
    df_all, missing = add_audio_features(df_all, raw_dir=args.raw_dir, sr=args.sr)

    # Cluster profile
    print("[step] cluster profile...")
    profile = make_cluster_profile(df_all)

    # Write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    out_utter = os.path.join(args.out_dir, "utterances_enriched.csv")
    out_prof = os.path.join(args.out_dir, "cluster_profile.csv")
    out_miss = os.path.join(args.out_dir, "missing_files.csv")

    df_all.to_csv(out_utter, index=False, encoding="utf-8-sig")
    profile.to_csv(out_prof, index=False, encoding="utf-8-sig")
    missing.to_csv(out_miss, index=False, encoding="utf-8-sig")

    print("[done] wrote:")
    print(" ", out_utter)
    print(" ", out_prof)
    print(" ", out_miss)
    print("\n[quick] cluster sizes:")
    print(df_all.groupby("cluster").size().reset_index(name="n").to_string(index=False))


if __name__ == "__main__":
    main()
