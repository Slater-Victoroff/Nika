#!/usr/bin/env python3
"""
Prepare the Big Buck Bunny benchmark frames using the sample video path exposed
by scikit-video.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import skvideo.datasets
except ImportError:  # pragma: no cover - runtime dependency check
    skvideo = None


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for Big Buck Bunny frame preparation.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Prepare Bunny PNG frame folders via scikit-video")
    parser.add_argument("--dataset-root", default="static/benchmarks/bunny")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=132)
    return parser.parse_args()


def ensure_runtime_dependencies(ffmpeg_bin: str) -> None:
    """Validate that the Bunny-preparation dependencies are available.

    Args:
        ffmpeg_bin: Name or path of the ffmpeg executable to require.
    """
    if skvideo is None:
        raise RuntimeError(
            "scikit-video is required for Bunny preparation. Install it with: python3 -m pip install scikit-video"
        )
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg not found on PATH: {ffmpeg_bin}")


def main() -> int:
    """Extract PNG benchmark frames from the sample Big Buck Bunny video.

    Returns:
        Process exit status code.
    """
    args = parse_args()
    ensure_runtime_dependencies(args.ffmpeg_bin)

    source_video = Path(skvideo.datasets.bigbuckbunny())
    if not source_video.is_file():
        raise FileNotFoundError(f"scikit-video did not provide a Bunny video file: {source_video}")

    dataset_root = Path(args.dataset_root)
    if list(dataset_root.glob("*.png")):
        print(f"Using existing frames in {dataset_root}")
        return 0

    dataset_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.ffmpeg_bin,
        "-y",
        "-i",
        str(source_video),
        "-vf",
        f"scale={args.width}:{args.height}",
        "-frames:v",
        str(args.frames),
        str(dataset_root / "%04d.png"),
    ]
    print(f"Using Bunny source video: {source_video}")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Prepared Bunny frames under {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
