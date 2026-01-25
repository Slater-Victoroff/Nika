"""
Train Nika models on all videos that don't have models yet.

USAGE:
    python train_all.py [--config CONFIG]
"""

import argparse
import glob
import os
import re

import torch

from load_data import load_video_frames
from nika import feature_test


def get_trained_videos(model_dir: str, config: str) -> set[str]:
    """Get set of video names that already have trained models."""
    pattern = f"{model_dir}/{config}-*-epoch*-psnr*.torch"
    trained = set()
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        # Extract video name from: {config}-{video}-epoch{N}-psnr{X.XX}.torch
        match = re.match(rf"^{config}-(\w+)-epoch\d+-psnr[\d.]+\.torch$", basename)
        if match:
            trained.add(match.group(1))
    return trained


def get_available_videos(video_dir: str) -> list[str]:
    """Get list of available video directories."""
    videos = []
    for entry in os.scandir(video_dir):
        if entry.is_dir():
            # Check it has PNG frames
            pngs = glob.glob(f"{entry.path}/*.png")
            if len(pngs) >= 100:  # At least 100 frames
                videos.append(entry.name)
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description="Train Nika models on all videos")
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
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models (default: models)",
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

    # Get videos to train
    available = get_available_videos(args.video_dir)
    trained = get_trained_videos(args.model_dir, args.config)

    print(f"Available videos: {available}")
    print(f"Already trained ({args.config}): {trained}")

    to_train = [v for v in available if v.lower() not in {t.lower() for t in trained}]
    print(f"Videos to train: {to_train}")

    if not to_train:
        print("All videos already have trained models!")
        return

    # Train each video
    for i, video_name in enumerate(to_train):
        print(f"\n{'='*60}")
        print(f"Training {i+1}/{len(to_train)}: {video_name}")
        print(f"{'='*60}\n")

        video_path = f"{args.video_dir}/{video_name}"
        vid = load_video_frames(
            video_path,
            device,
            max_frames=args.max_frames,
            dtype=torch.uint8,
            normalize=False,
        )
        print(f"Video shape: {vid.shape}")

        feature_test(vid, video_name, args.config, device=device)

        # Clear GPU memory between videos
        del vid
        torch.cuda.empty_cache()

    print("\nAll training complete!")


if __name__ == "__main__":
    main()
