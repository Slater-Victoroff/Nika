"""
Generate residual videos for Nika models.

Creates a side-by-side comparison video showing:
- Ground Truth: Original video frames
- Prediction: Model reconstruction
- Residual: Absolute difference (amplified for visibility)
"""

import argparse
import os

import imageio
import numpy as np
import torch

from load_data import load_video_frames
from model_loading import load_model, parse_model_filename
from nika import NikaBlock

# Mapping from model video names to actual folder names
VIDEO_NAME_MAP = {
    "beauty": ["beauty", "Beauty"],
    "bosphorus": ["bosphorus", "Bosphorus"],
    "honey": ["honey", "HoneyBee"],
    "jockey": ["jockey", "Jockey"],
    "ready": ["ready", "ReadySteadyGo", "ReadySetGo"],
    "shake": ["shake", "ShakeNDry"],
    "yacht": ["yacht", "YachtRide"],
}
def add_label_to_frame(frame: np.ndarray, labels: list[str]) -> np.ndarray:
    """Overlay text labels on the panels of a stitched comparison frame.

    Args:
        frame: Concatenated RGB comparison frame.
        labels: Panel titles to render across the top of the frame.

    Returns:
        The annotated frame array.
    """
    H, W, C = frame.shape
    panel_width = W // len(labels)
    label_height = 30
    font_scale = 0.8

    try:
        import cv2

        for i, label in enumerate(labels):
            x_start = i * panel_width
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_start, 0), (x_start + panel_width, label_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            cv2.putText(frame, label, (x_start + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    except ImportError:
        pass

    return frame


def generate_and_save_video(
    model: NikaBlock,
    video: torch.Tensor,
    device: str,
    output_path: str,
    fps: int = 30,
    batch_size: int = 10,
    amplification: float = 5.0,
) -> None:
    """Render a comparison video of ground truth, prediction, and residuals.

    Args:
        model: Restored model used to reconstruct frames.
        video: Source frame tensor shaped ``(T, C, H, W)``.
        device: Device used for inference.
        output_path: Destination path for the encoded MP4 file.
        fps: Frame rate to encode into the output video.
        batch_size: Number of frames to process per inference batch.
        amplification: Multiplier applied to residual magnitudes for visibility.
    """
    num_frames = video.shape[0]
    C, H, W = video.shape[1], video.shape[2], video.shape[3]

    labels = ["Ground Truth", "Prediction", f"Residual ({int(amplification)}x)"]
    num_batches = (num_frames + batch_size - 1) // batch_size

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    try:
        with torch.no_grad():
            for batch_idx in range(num_batches):
                min_t = batch_idx * batch_size
                max_t = min((batch_idx + 1) * batch_size, num_frames)

                batch_gt = video[min_t:max_t].to(torch.float32) / 255.0
                t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
                norm_t_batch = t_batch.float() / max(num_frames - 1, 1)
                prediction = model(norm_t_batch).clamp(0, 1)

                residual = (prediction - batch_gt).abs() * amplification
                residual = residual.clamp(0, 1)

                for i in range(prediction.shape[0]):
                    gt_frame = batch_gt[i].cpu().numpy()
                    pred_frame = prediction[i].cpu().numpy()
                    res_frame = residual[i].cpu().numpy()

                    combined = np.concatenate([gt_frame, pred_frame, res_frame], axis=2)
                    frame = (combined * 255).astype(np.uint8)
                    frame = frame.transpose(1, 2, 0)
                    frame = add_label_to_frame(frame, labels)
                    writer.append_data(frame)

                if min_t % 100 == 0:
                    print(f"Processed frames {min_t}-{max_t - 1}")
    finally:
        writer.close()

    print(f"Saved video to {output_path}")


def main():
    """Parse CLI arguments, load inputs, and emit a residual-comparison video."""
    parser = argparse.ArgumentParser(description="Generate residual comparison video")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="visuals", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames to process")
    parser.add_argument("--video-dir", type=str, default="static/benchmarks/uvg", help="Video directory")
    parser.add_argument("--amplification", type=float, default=5.0, help="Residual amplification")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config, video_name = parse_model_filename(args.model_path)
    print(f"Config: {config}, Video: {video_name}")

    candidate_names = VIDEO_NAME_MAP.get(video_name, [video_name])
    video_path = None
    for folder_name in candidate_names:
        candidate = os.path.join(args.video_dir, folder_name)
        if os.path.isdir(candidate):
            video_path = candidate
            break
    if video_path is None:
        raise FileNotFoundError(
            f"Could not find frames for {video_name} under {args.video_dir}. "
            f"Tried: {candidate_names}"
        )
    print(f"Loading video from {video_path}...")

    video = load_video_frames(video_path, device, max_frames=args.max_frames, dtype=torch.uint8, normalize=False)
    print(f"Video shape: {video.shape}")

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, video.shape, config, device)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{config}_{video_name}_comparison.mp4")

    print("Generating comparison video...")
    generate_and_save_video(model, video, device, output_path, fps=args.fps, batch_size=args.batch_size, amplification=args.amplification)

    print("Done!")


if __name__ == "__main__":
    main()
