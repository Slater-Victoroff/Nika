"""Checkpoint parsing and model reconstruction helpers for trained Nika variants."""

import os
import re

import torch

from ablations import NoConvNika, RealNika, TuckerNika, WeirdNika
from configs import REFERENCES
from nika import NikaBlock


# Mapping from model video names to actual folder names
VIDEO_NAME_MAP = {
    "honey": "HoneyBee",
    "jockey": "Jockey",
    "ready": "ReadySteadyGo",
    "shake": "ShakeNDry",
    "yacht": "YachtRide",
}


def parse_model_filename(path: str) -> tuple[str, str]:
    """
    Extract config name and video name from model filename.

    Args:
        path: Checkpoint path whose basename follows the training naming convention.

    Returns:
        A ``(config_name, video_name)`` tuple parsed from the checkpoint filename.
    """
    basename = os.path.basename(path)
    match = re.match(r"^(.+)-(\w+)-epoch\d+-psnr[\d.]+\.torch$", basename)
    if not match:
        raise ValueError(
            f"Could not parse model filename: {basename}. "
            "Expected pattern: {config}-{video}-epoch{N}-psnr{X.XX}.torch"
        )
    return match.group(1), match.group(2)


def _filter_kwargs(config_kwargs: dict, allowed: set[str]) -> dict:
    """Filter a configuration dictionary down to constructor-supported keys.

    Args:
        config_kwargs: Full configuration mapping for a model preset.
        allowed: Set of keys accepted by the target model constructor.

    Returns:
        A new dictionary containing only keys present in ``allowed``.
    """
    return {key: value for key, value in config_kwargs.items() if key in allowed}


def load_model(path: str, vid_shape: tuple, config: str, device: str) -> torch.nn.Module:
    """
    Load a trained model checkpoint and instantiate the matching architecture.

    Args:
        path: Filesystem path to the serialized checkpoint.
        vid_shape: Video tensor shape used to infer temporal and spatial dimensions.
        config: Configuration name used to choose the correct model class.
        device: Device on which to instantiate and load the model.

    Returns:
        An evaluation-mode torch module restored from the checkpoint.
    """
    if config not in REFERENCES:
        raise ValueError(f"Unknown config: {config}. Valid: {list(REFERENCES.keys())}")

    T, C, H, W = vid_shape
    config_kwargs = REFERENCES[config]

    model_cls = NikaBlock
    allowed_keys = {"base_grid_channels", "real_tucker_ranks", "complex_tucker_ranks", "conv_hidden"}

    if config == "real-small":
        model_cls = RealNika
        allowed_keys = {"base_grid_channels", "real_tucker_ranks", "conv_hidden"}
    elif config == "tucker-small":
        model_cls = TuckerNika
        allowed_keys = {"base_grid_channels", "real_tucker_ranks", "complex_tucker_ranks", "conv_hidden"}
    elif config == "weird-nika":
        model_cls = WeirdNika
        allowed_keys = {"base_grid_channels", "complex_tucker_ranks", "conv_hidden"}
    elif config == "noconv-nika":
        model_cls = NoConvNika
        allowed_keys = {"base_grid_channels", "real_tucker_ranks", "complex_tucker_ranks"}

    model_kwargs = _filter_kwargs(config_kwargs, allowed_keys)
    model = model_cls(
        target_shape=[4, H, W, T],
        k=4,
        **model_kwargs,
        out_channels=3,
        device=device,
    )

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
