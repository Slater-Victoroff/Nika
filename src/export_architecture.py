#!/usr/bin/env python3
"""
Export Nika model architecture to JSON for interactive visualization.
"""
import json
import torch
import torch.nn as nn
from nika import NikaBlock, RealTucker, ComplexTucker, FeatureGrid, BasicUpres, TuckerFactor
from configs import REFERENCES


def get_module_info(module, name="", depth=0):
    """Recursively serialize a module tree into a JSON-friendly dictionary.

    Args:
        module: Torch module to inspect.
        name: Display name to assign to the current node in the output tree.
        depth: Current nesting depth within the model hierarchy.

    Returns:
        A nested dictionary containing parameter counts, type metadata, and children.
    """
    info = {
        "name": name,
        "type": module.__class__.__name__,
        "params": sum(p.numel() for p in module.parameters(recurse=False)),
        "total_params": sum(p.numel() for p in module.parameters()),
        "children": [],
        "depth": depth,
    }

    # Add type-specific details
    if isinstance(module, nn.Conv2d):
        info["details"] = {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "groups": module.groups,
        }
    elif isinstance(module, nn.Linear):
        info["details"] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
    elif isinstance(module, nn.GroupNorm):
        info["details"] = {
            "num_groups": module.num_groups,
            "num_channels": module.num_channels,
        }
    elif isinstance(module, nn.PixelShuffle):
        info["details"] = {
            "upscale_factor": module.upscale_factor,
        }
    elif isinstance(module, TuckerFactor):
        info["details"] = {
            "target_dim": module.target_dim,
            "rank": module.rank,
            "is_complex": module.is_complex,
            "chunked": module.chunked,
        }
    elif isinstance(module, (RealTucker, ComplexTucker)):
        info["details"] = {
            "shape": [module.C, module.H, module.W, module.T],
            "ranks": [module.rC, module.rH, module.rW, module.rT],
        }
    elif isinstance(module, FeatureGrid):
        info["details"] = {
            "target_shape": [module.C, module.H, module.W, module.T],
            "grid_res": [module.grid_c, module.grid_h, module.grid_w, module.grid_t],
        }
    elif isinstance(module, BasicUpres):
        info["details"] = {
            "upscale_factor": module.k,
        }
    elif isinstance(module, NikaBlock):
        info["details"] = {
            "internal_shape": module.internal_shape,
        }

    # Recursively process children
    for child_name, child_module in module.named_children():
        child_info = get_module_info(child_module, child_name, depth + 1)
        info["children"].append(child_info)

    return info


def export_config(config_name, target_shape, k=4, device="cpu"):
    """Instantiate one model preset and export its architecture metadata.

    Args:
        config_name: Name of the preset to export from ``REFERENCES``.
        target_shape: Input tensor shape used to build the model.
        k: Spatial upscaling factor passed to ``NikaBlock``.
        device: Device used for the temporary model instantiation.

    Returns:
        A JSON-serializable dictionary describing the instantiated architecture.
    """
    config = REFERENCES.get(config_name, {})

    # Build kwargs, handling optional parameters
    model_kwargs = {}
    if "grid_ranks" in config:
        model_kwargs["grid_ranks"] = config["grid_ranks"]
    else:
        model_kwargs["grid_ranks"] = [0, 0, 0, 0]  # Will be skipped

    if "real_tucker_ranks" in config:
        model_kwargs["real_tucker_ranks"] = config["real_tucker_ranks"]
    else:
        model_kwargs["real_tucker_ranks"] = [0, 0, 0, 0]

    if "complex_tucker_ranks" in config:
        model_kwargs["complex_tucker_ranks"] = config["complex_tucker_ranks"]
    else:
        model_kwargs["complex_tucker_ranks"] = [0, 0, 0, 0]

    model_kwargs["conv_hidden"] = config.get("conv_hidden", 0)

    # Create model
    model = NikaBlock(
        target_shape=target_shape,
        k=k,
        **model_kwargs,
        out_channels=3,
        device=device,
    )

    # Extract architecture
    arch = get_module_info(model, "NikaBlock")
    arch["config_name"] = config_name
    arch["config"] = config
    arch["target_shape"] = target_shape
    arch["k"] = k

    return arch


def main():
    """Export the configured architecture presets to ``docs/data/architectures.json``."""
    # Standard video shape for visualization
    target_shape = [4, 1080, 1920, 600]  # C, H, W, T
    k = 4

    configs_to_export = ["small", "tucker-small", "real-small", "weird-nika", "noconv-nika"]

    all_architectures = {}

    for config_name in configs_to_export:
        print(f"Exporting {config_name}...")
        try:
            arch = export_config(config_name, target_shape, k, device="cpu")
            all_architectures[config_name] = arch
        except Exception as e:
            print(f"  Error: {e}")

    # Save to JSON
    output_path = "../docs/data/architectures.json"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_architectures, f, indent=2)

    print(f"\nSaved architectures to {output_path}")

    # Print summary
    for name, arch in all_architectures.items():
        print(f"\n{name}:")
        print(f"  Total params: {arch['total_params'] / 1e6:.3f}M")


if __name__ == "__main__":
    main()
