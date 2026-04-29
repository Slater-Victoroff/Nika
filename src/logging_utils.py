"""Logging helpers for model diagnostics."""


def log_nika_block_stats(model):
    """Print a parameter-count breakdown for a ``NikaBlock`` instance."""
    real_tucker_params = sum(p.numel() for p in model.real_tucker.parameters())
    complex_tucker_params = sum(p.numel() for p in model.complex_tucker.parameters())
    grid_params = sum(p.numel() for p in model.grid_features.parameters())
    upres_params = sum(p.numel() for p in model.upres.parameters())
    operator_params = sum(p.numel() for p in model.flow_operator.parameters())
    total_params = real_tucker_params + complex_tucker_params + grid_params + upres_params + operator_params

    print("NikaBlock parameters:")
    print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
    print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
    print(f"  Feature Grid:    {grid_params / 1e6:.3f}M")
    print(f"  Flow Operator:   {operator_params / 1e6:.3f}M")
    print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
    print(f"  Total:           {total_params / 1e6:.3f}M")
