#!/usr/bin/env python3
"""
Download the standard UVG benchmark archives from the official UVG website,
extract the raw YUV files, and convert them into PNG frame folders.

The training code expects:

    static/benchmarks/uvg/<video_name>/*.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    import py7zr
except ImportError:  # pragma: no cover - runtime dependency check
    py7zr = None


SEQUENCES = {
    "beauty": {
        "archive": "Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z",
        "url": "https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    },
    # "bosphorus": {
    #     "archive": "Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "url": "https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "width": 1920,
    #     "height": 1080,
    #     "fps": 120,
    # },
    # "honey": {
    #     "archive": "HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "url": "https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "width": 1920,
    #     "height": 1080,
    #     "fps": 120,
    # },
    # "jockey": {
    #     "archive": "Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "url": "https://ultravideo.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "width": 1920,
    #     "height": 1080,
    #     "fps": 120,
    # },
    # "ready": {
    #     "archive": "ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "url": "https://ultravideo.fi/video/ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "width": 1920,
    #     "height": 1080,
    #     "fps": 120,
    # },
    # "shake": {
    #     "archive": "ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "url": "https://ultravideo.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "width": 1920,
    #     "height": 1080,
    #     "fps": 120,
    # },
    # "yacht": {
    #     "archive": "YachtRide_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "url": "https://ultravideo.fi/video/YachtRide_1920x1080_120fps_420_8bit_YUV_RAW.7z",
    #     "width": 1920,
    #     "height": 1080,
    #     "fps": 120,
    # },
}


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for UVG dataset preparation.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Prepare UVG PNG frame folders")
    parser.add_argument("--dataset-root", default="static/benchmarks/uvg")
    parser.add_argument("--downloads-dir", default="static/downloads/uvg")
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=sorted(SEQUENCES.keys()),
        default=list(SEQUENCES.keys()),
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--keep-yuv", action="store_true")
    return parser.parse_args()


def ensure_runtime_dependencies(ffmpeg_bin: str) -> None:
    """Validate that extraction and conversion dependencies are available.

    Args:
        ffmpeg_bin: Name or path of the ffmpeg executable to require.
    """
    if py7zr is None:
        raise RuntimeError(
            "py7zr is required for UVG extraction. Install it with: python3 -m pip install py7zr"
        )
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg not found on PATH: {ffmpeg_bin}")


def download(url: str, destination: Path) -> None:
    """Download an archive unless it already exists on disk.

    Args:
        url: Remote archive URL to download.
        destination: Local path where the archive should be stored.
    """
    if destination.exists():
        print(f"Using existing archive: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract_yuv(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a downloaded UVG archive and return the contained YUV file path.

    Args:
        archive_path: Path to the downloaded ``.7z`` archive.
        extract_dir: Directory where the archive contents should be expanded.

    Returns:
        Path to the extracted raw YUV video file.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)
    yuv_files = sorted(extract_dir.rglob("*.yuv"))
    if not yuv_files:
        raise FileNotFoundError(f"No .yuv file found in {archive_path}")
    return yuv_files[0]


def yuv_to_png(yuv_path: Path, frame_dir: Path, width: int, height: int, fps: int, ffmpeg_bin: str) -> None:
    """Convert one raw YUV sequence into the PNG frame layout expected by training.

    Args:
        yuv_path: Raw YUV input file.
        frame_dir: Output directory for numbered PNG frames.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate metadata passed to ffmpeg.
        ffmpeg_bin: Name or path of the ffmpeg executable to run.
    """
    if list(frame_dir.glob("*.png")):
        print(f"Using existing frames in {frame_dir}")
        return
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        str(yuv_path),
        str(frame_dir / "%04d.png"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    """Download, extract, and convert the selected UVG sequences.

    Returns:
        Process exit status code.
    """
    args = parse_args()
    ensure_runtime_dependencies(args.ffmpeg_bin)

    dataset_root = Path(args.dataset_root)
    downloads_root = Path(args.downloads_dir)

    for name in args.sequences:
        meta = SEQUENCES[name]
        archive_path = downloads_root / meta["archive"]
        extract_dir = downloads_root / name
        frame_dir = dataset_root / name

        download(meta["url"], archive_path)
        yuv_path = extract_yuv(archive_path, extract_dir)
        yuv_to_png(
            yuv_path=yuv_path,
            frame_dir=frame_dir,
            width=meta["width"],
            height=meta["height"],
            fps=meta["fps"],
            ffmpeg_bin=args.ffmpeg_bin,
        )

        if not args.keep_yuv and extract_dir.exists():
            shutil.rmtree(extract_dir)

    print(f"Prepared UVG frames under {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
