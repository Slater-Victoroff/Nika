"""
Train Nika models on videos.

USAGE:
    python train.py <video_name> [--config CONFIG]

EXAMPLE:
    python train.py HoneyBee --config small
"""

import argparse
import torch

from load_data import load_video_frames
from nika import feature_test


def main():
    parser = argparse.ArgumentParser(description="Train Nika model on a video")
    parser.add_argument("video_name", type=str, help="Name of the video directory")
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["xxs", "xs", "small", "medium", "large"],
        help="Model configuration (default: small)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="static/benchmarks",
        help="Base directory for video frames (default: static/benchmarks)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum number of frames to use (default: 600)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    args = parser.parse_args()

    # Setup device and optimizations
    device = args.device
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load video
    video_path = f"{args.video_dir}/{args.video_name}"
    print(f"Loading video from {video_path}...")
    vid = load_video_frames(
        video_path,
        device,
        max_frames=args.max_frames,
        dtype=torch.uint8,
        normalize=False,
    )
    print(f"Video shape: {vid.shape}")

    # Train
    print(f"Training {args.config} model on {args.video_name}...")
    feature_test(vid, args.video_name, args.config, device=device)


if __name__ == "__main__":
    main()
