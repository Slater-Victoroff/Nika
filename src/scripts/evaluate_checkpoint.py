#!/usr/bin/env python3
"""
Evaluate a checkpoint against a PNG frame folder and report average PSNR plus
decode FPS. This keeps the logic local to the script so the main source tree
does not need to be patched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI arguments for checkpoint evaluation.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate one NIKA checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-root", default="static/benchmarks/uvg")
    parser.add_argument("--video", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--json", dest="json_path", default=None)
    return parser.parse_args()


def parse_checkpoint_name(checkpoint: Path) -> tuple[str, str]:
    """Infer the config and video name from a checkpoint filename.

    Args:
        checkpoint: Checkpoint path whose basename follows the training naming scheme.

    Returns:
        A ``(config_name, video_name)`` tuple extracted from the filename.
    """
    match = re.match(r"^(.+)-(\w+)-epoch\d+-psnr[\d.]+\.torch$", checkpoint.name)
    if not match:
        raise ValueError(f"Could not parse checkpoint name: {checkpoint.name}")
    return match.group(1), match.group(2)


def load_model(checkpoint: Path, video_shape: tuple[int, int, int, int], config: str, device: str) -> NikaBlock:
    """Instantiate and restore the model needed to evaluate one checkpoint.

    Args:
        checkpoint: Checkpoint file to restore.
        video_shape: Source video shape used to size the model.
        config: Model preset name to instantiate.
        device: Device on which to load the model.

    Returns:
        An evaluation-mode ``NikaBlock`` loaded with checkpoint weights.
    """
    if config not in REFERENCES:
        raise ValueError(f"Unknown config: {config}")
    t, c, h, w = video_shape
    model = NikaBlock(
        target_shape=[4, h, w, t],
        k=4,
        **REFERENCES[config],
        out_channels=3,
        device=device,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> int:
    """Run the checkpoint evaluation workflow and emit metrics as JSON.

    Returns:
        Process exit status code.
    """
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    config, inferred_video = parse_checkpoint_name(checkpoint)
    video_name = args.video or inferred_video
    video_dir = Path(args.dataset_root) / video_name
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {video_dir}")

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
    model = load_model(checkpoint, tuple(video.shape), config, args.device)

    frame_count = int(video.shape[0])
    batch_size = int(args.batch_size)
    total_psnr = 0.0

    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            target = video[start:end].to(torch.float32) / 255.0
            idx = torch.arange(start, end, device=args.device, dtype=torch.int64)
            norm_t = idx.float() / max(frame_count - 1, 1)
            prediction = model(norm_t).clamp(0, 1)
            mse = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.sum().item()

    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            idx = torch.arange(start, end, device=args.device, dtype=torch.int64)
            norm_t = idx.float() / max(frame_count - 1, 1)
            _ = model(norm_t)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - start_time

    metrics = {
        "checkpoint": str(checkpoint),
        "config": config,
        "video": video_name,
        "frames": frame_count,
        "avg_psnr": total_psnr / frame_count,
        "decode_fps": frame_count / elapsed if elapsed > 0 else None,
        "elapsed_seconds": elapsed,
        "device": args.device,
    }

    print(json.dumps(metrics, indent=2))
    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
