import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

RAW_DIR = "/hpctmp/scratch/e0968015/mosei/Raw"

AUDIO_WINS_CSV = "/hpctmp/scratch/e0968015/mosei/FusionRuns/posthoc_audio_vs_vision_run3/mosei_A35_top_audio_wins.csv"
VISION_WINS_CSV = "/hpctmp/scratch/e0968015/mosei/FusionRuns/posthoc_audio_vs_vision_run3/mosei_A35_top_vision_wins.csv"

OUT_CSV = "/hpctmp/scratch/e0968015/mosei/FusionRuns/posthoc_audio_vs_vision_run3/mosei_A35_topwins_audio_prosody.csv"

# NEW: summary outputs
OUT_SUMMARY_CSV = "/hpctmp/scratch/e0968015/mosei/FusionRuns/posthoc_audio_vs_vision_run3/mosei_A35_topwins_audio_prosody_summary.csv"


def parse_id(id_str: str):
    s = str(id_str)
    if "$_$" not in s:
        return None, None
    vid, clip = s.split("$_$", 1)
    return vid, clip


def video_path_from_id(videoid: str, clipid: str) -> str:
    return str(Path(RAW_DIR) / videoid / f"{clipid}.mp4")


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

    # fallback (only works if librosa backend can decode the container)
    y, _ = librosa.load(mp4_path, sr=sr, mono=True)
    return y


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


def load_topwins(csv_path: str, group_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["group"] = group_label

    vids, clips, paths = [], [], []
    for x in df["id"].astype(str).tolist():
        vid, clip = parse_id(x)
        vids.append(vid)
        clips.append(clip)
        paths.append(video_path_from_id(vid, clip) if vid and clip else None)

    df["video_id"] = vids
    df["clip_id"] = clips
    df["video_path"] = paths
    return df


def make_summary(out: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise prosody features by (group, sentiment_class).
    """
    feats = [
        "f0_mean_hz", "f0_std_hz", "f0_range_hz", "voiced_ratio",
        "rms_mean", "rms_std", "pause_ratio", "duration_s"
    ]
    # keep only rows with extracted features (no extraction error)
    df = out.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()

    # NEW: joint grouping key
    df["group_sentiment"] = df["group"].astype(str) + "__" + df["sentiment_class"].astype(str)

    rows = []
    for (g, s), sub in df.groupby(["group", "sentiment_class"], dropna=False):
        rec = {
            "group": g,
            "sentiment_class": s,
            "n": int(len(sub)),
        }
        for c in feats:
            if c in sub.columns:
                rec[f"{c}_mean"] = float(np.nanmean(sub[c].to_numpy(dtype=float)))
                rec[f"{c}_std"] = float(np.nanstd(sub[c].to_numpy(dtype=float)))
        rows.append(rec)

    return pd.DataFrame(rows).sort_values(["sentiment_class", "group"])


def main():
    audio_df = load_topwins(AUDIO_WINS_CSV, "audio_win")
    vision_df = load_topwins(VISION_WINS_CSV, "vision_win")
    df = pd.concat([audio_df, vision_df], ignore_index=True)

    df["exists"] = df["video_path"].apply(lambda p: bool(p) and os.path.exists(p))

    print("[info] total rows:", len(df))
    print("[info] missing files:", int((~df["exists"]).sum()))

    rows = []
    sr = 16000

    for i, r in df.iterrows():
        if not r["exists"]:
            rows.append({**r.to_dict(), "error": "missing_video_file"})
            continue

        mp4 = r["video_path"]
        try:
            y = load_audio_from_mp4(mp4, sr=sr)
            feats = compute_prosody_features(y, sr=sr)
            rows.append({**r.to_dict(), **feats})
        except Exception as e:
            rows.append({**r.to_dict(), "error": str(e)})

        if (i + 1) % 50 == 0:
            print(f"[progress] {i+1}/{len(df)} processed", flush=True)

    out = pd.DataFrame(rows)

    # NEW: joint label for easier filtering later
    out["group_sentiment"] = out["group"].astype(str) + "__" + out["sentiment_class"].astype(str)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("[done] wrote:", OUT_CSV)

    summary = make_summary(out)
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print("[done] wrote:", OUT_SUMMARY_CSV)

    # Quick print of counts by group/sentiment (useful sanity check)
    print("\n[counts] by group x sentiment_class:")
    print(out.groupby(["group", "sentiment_class"]).size().reset_index(name="n").to_string(index=False))


if __name__ == "__main__":
    main()
