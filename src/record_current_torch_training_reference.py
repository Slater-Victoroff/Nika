from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock
from soap import SOAP


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any("_orig_mod" in key for key in state):
        return state
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = key
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        clean_key = clean_key.replace("._orig_mod.", ".")
        cleaned[clean_key] = value
    return cleaned


def _tensor_dict_to_numpy(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in sorted(tensors.items())}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters()}


def _parse_frame_indices(raw: str | None, n_frames: int, num_steps: int) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    indices = np.linspace(0, n_frames - 1, num=num_steps, dtype=np.int64)
    return [int(item) for item in indices.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-frames", type=int, default=132)
    parser.add_argument("--frame-indices")
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--betas", default="0.95,0.95")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--max-precond-dim", type=int, default=10000)
    parser.add_argument("--merge-dims", action="store_true")
    parser.add_argument("--precondition-1d", action="store_true")
    parser.add_argument("--normalize-grads", action="store_true")
    parser.add_argument("--correct-bias", action="store_true", default=True)
    parser.add_argument("--no-correct-bias", action="store_false", dest="correct_bias")
    args = parser.parse_args()

    beta1, beta2 = (float(item.strip()) for item in args.betas.split(","))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    vid = load_video_frames(
        args.frame_dir,
        device=args.device,
        max_frames=args.n_frames,
        dtype=torch.uint8,
        normalize=False,
    )

    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **REFERENCES[args.config],
        out_channels=3,
        device=args.device,
    )
    state = _clean_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.load_state_dict(state)
    model.train()

    optimizer = SOAP(
        model.parameters(),
        lr=args.lr,
        betas=(beta1, beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        precondition_frequency=args.precondition_frequency,
        max_precond_dim=args.max_precond_dim,
        merge_dims=args.merge_dims,
        precondition_1d=args.precondition_1d,
        normalize_grads=args.normalize_grads,
        correct_bias=args.correct_bias,
    )

    frame_indices = _parse_frame_indices(args.frame_indices, int(vid.shape[0]), args.num_steps)
    norm_t = np.asarray(frame_indices, dtype=np.float32) / float(vid.shape[0] - 1)
    targets = (vid[frame_indices].to(torch.float32) / 255.0).detach().cpu().numpy()

    _save_npz(out_dir / "initial_params.npz", _tensor_dict_to_numpy(model.state_dict()))
    np.save(out_dir / "frame_indices.npy", np.asarray(frame_indices, dtype=np.int64))
    np.save(out_dir / "norm_t.npy", norm_t)
    np.save(out_dir / "targets.npy", targets)

    params_by_name = _named_parameters(model)
    metadata = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": args.device,
        "n_frames": int(vid.shape[0]),
        "frame_indices": frame_indices,
        "optimizer": {
            "lr": args.lr,
            "betas": [beta1, beta2],
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "precondition_frequency": args.precondition_frequency,
            "max_precond_dim": args.max_precond_dim,
            "merge_dims": args.merge_dims,
            "precondition_1d": args.precondition_1d,
            "normalize_grads": args.normalize_grads,
            "data_format": "channels_first",
            "correct_bias": args.correct_bias,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for step_idx, frame_idx in enumerate(frame_indices):
        target = vid[frame_idx].to(torch.float32) / 255.0
        step_dir = out_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        optimizer.zero_grad(set_to_none=True)
        current_norm_t = torch.tensor([norm_t[step_idx]], device=args.device, dtype=torch.float32)
        prediction = model(current_norm_t)
        mse = F.mse_loss(prediction.squeeze(0), target)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        loss = (-psnr).mean()
        loss.backward()

        gradients = {
            name: param.grad.detach().cpu().numpy()
            for name, param in sorted(params_by_name.items())
            if param.grad is not None
        }
        _save_npz(step_dir / "gradients.npz", gradients)
        np.save(step_dir / "prediction.npy", prediction.detach().cpu().numpy())

        optimizer.step()
        _save_npz(step_dir / "post_params.npz", _tensor_dict_to_numpy(model.state_dict()))

        metrics = {
            "step": step_idx,
            "frame_idx": frame_idx,
            "norm_t": float(norm_t[step_idx]),
            "loss": float(loss.item()),
            "mse": float(mse.item()),
            "psnr": float(psnr.item()),
        }
        (step_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics))

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
