"""
Visualize the learned FeatureGrid from a trained Nika model.

The FeatureGrid is a 4D learned tensor (C, H, W, T) that captures
spatial-temporal features. This script creates visualizations showing:
1. Grid of all channels at a specific time slice
2. Evolution of a channel across time
3. Combined overview image
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

from model_loading import parse_model_filename


def load_feature_grid(path: str, device: str) -> torch.Tensor:
    """Load the serialized feature-grid parameter tensor from a checkpoint.

    Args:
        path: Path to the checkpoint file to inspect.
        device: Device used when deserializing the checkpoint.

    Returns:
        The saved ``grid_features.grid`` tensor.
    """
    state_dict = torch.load(path, map_location=device)
    grid = state_dict["grid_features.grid"]
    return grid


def normalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
    """Scale a tensor into the display-friendly ``[0, 1]`` interval.

    Args:
        tensor: Tensor to normalize for image export.

    Returns:
        A normalized tensor suitable for visualization.
    """
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min > 1e-8:
        return (tensor - t_min) / (t_max - t_min)
    return tensor - t_min


def visualize_feature_grid(grid: torch.Tensor, output_dir: str, name: str):
    """Render several summary visualizations for a learned feature grid.

    Args:
        grid: Feature-grid tensor shaped ``(C, H, W, T)``.
        output_dir: Directory where the generated images should be written.
        name: Prefix used when naming the exported visualization files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Grid shape: (C, H, W, T)
    grid = grid.detach().cpu()
    C, H, W, T = grid.shape
    print(f"FeatureGrid shape: C={C}, H={H}, W={W}, T={T}")

    # 1. Visualize all channels at middle time slice
    t_mid = T // 2
    channels_at_t = grid[:, :, :, t_mid]  # (C, H, W)
    channels_norm = normalize_for_display(channels_at_t)

    # Create grid image of all channels
    # Add channel dimension for make_grid: (C, 1, H, W)
    channels_for_grid = channels_norm.unsqueeze(1)
    nrow = int(np.ceil(np.sqrt(C)))
    grid_img = make_grid(channels_for_grid, nrow=nrow, normalize=False, padding=2)
    save_image(grid_img, os.path.join(output_dir, f"{name}_channels_t{t_mid}.png"))
    print(f"Saved channel grid at t={t_mid}")

    # 2. Visualize temporal evolution for first few channels (skip if T=1)
    if T > 1:
        num_time_samples = min(8, T)
        time_indices = torch.linspace(0, T - 1, num_time_samples).long()

        fig, axes = plt.subplots(min(4, C), num_time_samples, figsize=(2 * num_time_samples, 2 * min(4, C)))
        if min(4, C) == 1:
            axes = axes.reshape(1, -1)
        if num_time_samples == 1:
            axes = axes.reshape(-1, 1)

        for c in range(min(4, C)):
            for i, t in enumerate(time_indices):
                ax = axes[c, i]
                slice_data = normalize_for_display(grid[c, :, :, t]).numpy()
                ax.imshow(slice_data, cmap='viridis', aspect='auto')
                ax.set_title(f'C{c}, t={t.item()}', fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_temporal_evolution.png"), dpi=150)
        plt.close()
        print(f"Saved temporal evolution plot")
    else:
        print(f"Skipping temporal evolution (T=1)")

    # 3. Create mean activation over time for each channel
    mean_over_time = grid.mean(dim=3)  # (C, H, W)
    mean_norm = normalize_for_display(mean_over_time)
    mean_for_grid = mean_norm.unsqueeze(1)
    mean_grid_img = make_grid(mean_for_grid, nrow=nrow, normalize=False, padding=2)
    save_image(mean_grid_img, os.path.join(output_dir, f"{name}_channels_mean.png"))
    print(f"Saved mean activation grid")

    # 4. Create std (variance) over time - shows which areas change most
    std_over_time = grid.std(dim=3)  # (C, H, W)
    std_norm = normalize_for_display(std_over_time)
    std_for_grid = std_norm.unsqueeze(1)
    std_grid_img = make_grid(std_for_grid, nrow=nrow, normalize=False, padding=2)
    save_image(std_grid_img, os.path.join(output_dir, f"{name}_channels_std.png"))
    print(f"Saved temporal std grid")

    # 5. Summary statistics
    print(f"\nFeatureGrid Statistics:")
    print(f"  Min: {grid.min().item():.4f}")
    print(f"  Max: {grid.max().item():.4f}")
    print(f"  Mean: {grid.mean().item():.4f}")
    print(f"  Std: {grid.std().item():.4f}")


def main():
    """Parse CLI arguments and generate feature-grid visualizations for one checkpoint."""
    parser = argparse.ArgumentParser(description="Visualize FeatureGrid from trained model")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="visuals", help="Output directory")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config, video_name = parse_model_filename(args.model_path)
    print(f"Config: {config}, Video: {video_name}")

    print(f"Loading FeatureGrid from {args.model_path}...")
    grid = load_feature_grid(args.model_path, device)

    visualize_feature_grid(grid, args.output_dir, f"{config}_{video_name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
