import csv
import os
import subprocess
from pathlib import Path

LABEL_CSV = "/hpctmp/scratch/e0968015/chsims/label.csv"
RAW_ROOT = Path("/hpctmp/scratch/e0968015/chsims/Raw")
WAV_ROOT = Path("/hpctmp/scratch/e0968015/chsims/wav")

FFMPEG = "ffmpeg"
AUDIO_SR = 16000
AUDIO_CH = 1

WAV_ROOT.mkdir(parents=True, exist_ok=True)


def extract_audio(mp4_path: Path, wav_path: Path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-i", str(mp4_path),
        "-vn",
        "-ac", str(AUDIO_CH),
        "-ar", str(AUDIO_SR),
        "-acodec", "pcm_s16le",
        str(wav_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mp4_path}\n{result.stderr}")


def main():
    num_ok = 0
    num_fail = 0

    with open(LABEL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=1):
            video_id = row["video_id"].strip()
            clip_id = row["clip_id"].strip()

            mp4_path = RAW_ROOT / video_id / f"{clip_id}.mp4"
            wav_path = WAV_ROOT / video_id / f"{clip_id}.wav"

            if not mp4_path.exists():
                print(f"[{row_idx}] Missing MP4: {mp4_path}")
                num_fail += 1
                continue

            if wav_path.exists():
                print(f"[{row_idx}] Already exists, skipping: {wav_path}")
                num_ok += 1
                continue

            try:
                extract_audio(mp4_path, wav_path)
                print(f"[{row_idx}] OK: {wav_path}")
                num_ok += 1
            except Exception as e:
                print(f"[{row_idx}] FAILED: {mp4_path}")
                print(str(e))
                num_fail += 1

    print(f"Done. Success={num_ok}, Fail={num_fail}")


if __name__ == "__main__":
    main()
