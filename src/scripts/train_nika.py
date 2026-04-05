#!/usr/bin/env python3
"""
Thin CLI wrapper around the existing training entrypoint in src/nika.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from load_data import load_video_frames
from nika import feature_test


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI arguments for single-sequence training.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Train NIKA on one frame folder")
    parser.add_argument("--dataset-root", default="static/benchmarks/uvg")
    parser.add_argument("--video", required=True, help="Frame folder name, e.g. beauty")
    parser.add_argument("--config", required=True, help="Model config, e.g. small/medium/large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    """Load one dataset folder and invoke the existing training entrypoint.

    Returns:
        Process exit status code.
    """
    args = parse_args()
    video_dir = Path(args.dataset_root) / args.video
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {video_dir}")

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    if str(args.device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video = load_video_frames(
        str(video_dir),
        device=args.device,
        max_frames=args.max_frames,
        dtype=torch.uint8,
        normalize=False,
    )
    feature_test(video, args.video, args.config, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
