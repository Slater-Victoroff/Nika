"""Reference hyperparameter presets used to instantiate Nika model variants."""

REFERENCES = {
    "xxs": {
        "base_grid_channels": 1,
        "real_tucker_ranks": [2, 45, 45, 45],
        "complex_tucker_ranks": [2, 40, 40, 40],
        "conv_hidden": 48,
    },
    "xs": {
        "base_grid_channels": 2,
        "real_tucker_ranks": [3, 50, 50, 50],
        "complex_tucker_ranks": [3, 45, 45, 45],
        "conv_hidden": 48,
    },
    "small": {
        "base_grid_channels": 4,
        "real_tucker_ranks": [3, 75, 75, 60],
        "complex_tucker_ranks": [3, 60, 60, 50],
        "conv_hidden": 48,
    },
    "medium": {
        "base_grid_channels": 8,
        "real_tucker_ranks": [4, 85, 85, 70],
        "complex_tucker_ranks": [4, 65, 65, 55],
        "conv_hidden": 48,
    },
    "large": {
        "base_grid_channels": 16,
        "real_tucker_ranks": [4, 100, 100, 100],
        "complex_tucker_ranks": [4, 90, 90, 80],
        "conv_hidden": 48,
    },
}
