from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    arrays = {key: value.detach().cpu().numpy() for key, value in state.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    print(output)
    print(len(arrays))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
