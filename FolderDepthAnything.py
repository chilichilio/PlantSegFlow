#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import List


def run(cmd):
    print("\n▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def split_video_1min(video: Path, segment_sec: int = 60) -> list[Path]:
    out_dir = video.parent
    prefix = video.stem
    pattern = out_dir / f"{prefix}_%03d.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-f", "segment",
        "-segment_time", str(segment_sec),
        "-c", "copy",
        str(pattern),
    ]
    run(cmd)

    clips = []
    for fp in sorted(out_dir.glob(f"{prefix}_*.mp4")):
        # prefix_001.mp4 -> prefix_1.mp4
        tail = fp.stem.split("_")[-1]
        try:
            idx = int(tail)  # 001 -> 1
        except ValueError:
            continue
        new_fp = out_dir / f"{prefix}_{idx}.mp4"
        if fp != new_fp and not new_fp.exists():
            fp.rename(new_fp)
        clips.append(new_fp)

    clips = sorted(clips, key=lambda p: int(p.stem.split("_")[-1]))
    print(f"✅ Split into {len(clips)} clip(s)")
    return clips


def run_vda(clips: List[Path], grayscale: bool, encoder: str, video_name: str):
    run_py = Path(__file__).parent / "run.py"
    if not run_py.exists():
        raise FileNotFoundError("run.py not found in the same folder")

    # output folder = video name (no extension)
    out_root = Path(__file__).parent / video_name
    out_root.mkdir(exist_ok=True)

    for mp4 in clips:
        cmd = [
            "python3", str(run_py),
            "--input_video", str(mp4),
            "--output_dir", str(out_root),
            "--encoder", encoder,
        ]
        if grayscale:
            cmd.append("--grayscale")

        print(f"\n▶ VideoDepthAnything: {mp4.name}")
        run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="MVI_7208.mp4",
        help="Main video filename (default: MVI_7208.mp4)"
    )
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    clips = split_video_1min(video)
    run_vda(clips,grayscale=args.grayscale,encoder=args.encoder,video_name=video.stem)


if __name__ == "__main__":
    main()

