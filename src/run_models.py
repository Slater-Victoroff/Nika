import glob
import math
import copy
import os
import re
import time

import torch
import torch.nn.functional as F
import argparse

from load_data import load_video_frames
from nika import MosaicNika
from soap import SOAP
from configs import REFERENCES

try:
    from fvcore.nn.flop_count import FlopCountAnalysis as FlopCounterMode
except Exception:
    FlopCounterMode = None


def get_best_model(model_dir, vid_shape, vid_name, config, device, compile_ops=False):
    model_vid = os.path.basename(vid_name)
    candidates = glob.glob(f"{model_dir}/{config}-{model_vid}-*.torch")

    def extract_psnr(path):
        match = re.search(r'psnr([0-9]+(?:\.[0-9]+)?)', path)
        if match:
            return float(match.group(1))
        return float('-inf')

    actual_config = config
    if not candidates:
        fallback = glob.glob(f"{model_dir}/*-{model_vid}-*.torch")
        if not fallback:
            raise RuntimeError(f"No models found for {vid_name} with config {config} in {model_dir}")
        fallback.sort(key=extract_psnr)
        model_path = fallback[-1]
        prefix = os.path.basename(model_path).split(f"-{model_vid}-")[0]
        if prefix not in REFERENCES:
            raise RuntimeError(
                f"Found model file {model_path} but inferred prefix '{prefix}' is not in REFERENCES"
            )
        actual_config = prefix
        print(f"Info: using config '{actual_config}' inferred from {model_path}")
    else:
        candidates.sort(key=extract_psnr)
        model_path = candidates[-1]
        print(f"Best model for {vid_name} with config {config}: {model_path}")

    model_kwargs = dict(REFERENCES[actual_config])
    num_segments = model_kwargs.pop('num_segments', 1)
    if os.path.basename(vid_name) == 'bunny' and 'grid_ranks' in model_kwargs:
        model_kwargs['grid_ranks'] = model_kwargs['grid_ranks'] * 2

    model = MosaicNika(
        target_shape=[4, vid_shape[2], vid_shape[3], vid_shape[0]],
        k=4,
        model_kwargs=model_kwargs,
        out_channels=3,
        device=device,
        num_segments=num_segments,
        compile_ops=compile_ops,
    )

    state_dict = torch.load(model_path, map_location=device)
    ckpt_map = {k.replace('._orig_mod', ''): v for k, v in state_dict.items()}
    mapped_sd = {}
    for mk in model.state_dict().keys():
        norm_mk = mk.replace('._orig_mod', '')
        if norm_mk in ckpt_map:
            mapped_sd[mk] = ckpt_map[norm_mk]

    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    if missing:
        print(f"Warning: missing keys when loading {model_path}: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading {model_path}: {unexpected}")
    return model


def compute_macs(model, device):
    if FlopCounterMode is None:
        return None

    sample_model = model
    if hasattr(model, 'models') and len(model.models) > 0:
        sample_model = model.models[0]
    while hasattr(sample_model, '_orig_mod'):
        sample_model = sample_model._orig_mod

    sample_t = torch.tensor([0.0], device=device, dtype=torch.float32)
    try:
        flops = FlopCounterMode(sample_model, (sample_t,))
        return flops.total()
    except Exception as exc:
        print(f"Warning: could not compute MACs: {exc}")
        return None


def warmup_model(model, num_frames, device, batch_size=8, repeats=10):
    model.eval()
    use_cuda = 'cuda' in device
    if use_cuda:
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(repeats):
            for start_idx in range(0, num_frames, batch_size):
                end_idx = min(num_frames, start_idx + batch_size)
                norm_t = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32) / max(num_frames - 1, 1)
                _ = model(norm_t)
    if use_cuda:
        torch.cuda.synchronize(device)


def measure_decode_fps(model, num_frames, device, batch_size=8, repeats=20):
    model.eval()
    times = torch.linspace(0.0, 1.0, steps=num_frames, device=device, dtype=torch.float32)
    use_cuda = 'cuda' in device

    with torch.no_grad():
        for _ in range(10):
            for start_idx in range(0, num_frames, batch_size):
                end_idx = min(num_frames, start_idx + batch_size)
                _ = model(times[start_idx:end_idx])

    if use_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            for start_idx in range(0, num_frames, batch_size):
                end_idx = min(num_frames, start_idx + batch_size)
                _ = model(times[start_idx:end_idx])

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total_frames = float(num_frames * repeats)
    return total_frames / elapsed if elapsed > 0 else float('inf')


def measure_psnr(model, vid, device, batch_size=8, debug_psnr=False):
    model.eval()
    num_frames = vid.shape[0]
    total_psnr = 0.0
    total_frames = 0

    with torch.no_grad():
        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(num_frames, start_idx + batch_size)
            norm_t = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32) / max(num_frames - 1, 1)
            pred = model(norm_t)
            pred_clamped = pred.clamp(0.0, 1.0)
            gt = vid[start_idx:end_idx].to(torch.float32) / 255.0

            per_frame_mse = F.mse_loss(pred_clamped, gt, reduction='none')
            per_frame_mse = per_frame_mse.view(per_frame_mse.shape[0], -1).mean(dim=1)
            frame_psnr = 10.0 * torch.log10(1.0 / (per_frame_mse + 1e-8))
            if debug_psnr:
                for idx, psnr_value in enumerate(frame_psnr.tolist(), start=start_idx):
                    print(f"frame {idx}: psnr={psnr_value:.6f}")
            total_psnr += frame_psnr.sum().item()
            total_frames += frame_psnr.shape[0]

    avg_psnr = total_psnr / total_frames if total_frames > 0 else 0.0
    return float(avg_psnr)


def measure_encode_fps(model, vid, device, steps=5, n_frames=None, batch_size=8):
    try:
        encode_model = copy.deepcopy(model)
    except Exception:
        encode_model = copy.deepcopy(getattr(model, '_orig_mod', model))

    encode_model.train()
    if n_frames is None:
        n_frames = vid.shape[0]
    n_frames = min(n_frames, vid.shape[0])
    batch_size = min(batch_size, n_frames)
    batch_ranges = [(i, min(n_frames, i + batch_size)) for i in range(0, n_frames, batch_size)]
    opt = SOAP(encode_model.parameters(), lr=1e-2)

    use_cuda = 'cuda' in device
    if use_cuda:
        torch.cuda.synchronize(device)
    else:
        pass

    norm_t_batches = []
    target_batches = []
    use_target_device = vid.device == torch.device(device) if isinstance(device, str) else vid.device == device
    for start_idx, end_idx in batch_ranges:
        idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32)
        norm_t_batches.append(idx / max(vid.shape[0] - 1, 1))
        target = vid[start_idx:end_idx].float() / 255.0
        if not use_target_device:
            target = target.to(device)
        target_batches.append(target)

    warmup_steps = min(2, steps)
    for _ in range(warmup_steps):
        for norm_t, target in zip(norm_t_batches, target_batches):
            opt.zero_grad(set_to_none=True)
            pred = encode_model(norm_t)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

    if use_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(steps):
        for norm_t, target in zip(norm_t_batches, target_batches):
            opt.zero_grad(set_to_none=True)
            pred = encode_model(norm_t)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total_frames = float(n_frames * steps)
    return total_frames / elapsed if elapsed > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description="Run Nika decode/encode benchmark")
    parser.add_argument("--basedir", default="static/benchmarks", help="Base video directory")
    parser.add_argument("--name", required=True, help="Video name (e.g., uvg/beauty)")
    parser.add_argument("--config", default="large", help="Config name from configs.REFERENCES")
    parser.add_argument("--device", default="cuda:0", help="Device to run on, e.g. cuda:0 or cpu")
    parser.add_argument("--model_dir", default="models/ref_models/large", help="Directory containing reference model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for decode/PSNR measurement")
    parser.add_argument("--compile_ops", action="store_true", help="Compile Nika internal operator modules for inference")
    parser.add_argument("--compile_model", action="store_true", help="Compile the whole loaded MosaicNika model for inference")
    parser.add_argument("--debug_psnr", action="store_true", help="Print per-frame PSNR values during decode evaluation")
    args = parser.parse_args()

    device = args.device
    if isinstance(device, str):
        if re.fullmatch(r"\d+", device):
            device = f"cuda:{device}"
        elif re.fullmatch(r"cuda\d+", device):
            device = device.replace('cuda', 'cuda:')

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vid = load_video_frames(f"{args.basedir}/{args.name}", device, dtype=torch.uint8, normalize=False)
    compile_ops = args.compile_ops and not args.compile_model
    model = get_best_model(args.model_dir, vid.shape, args.name, args.config, device, compile_ops=compile_ops)

    if args.compile_model:
        try:
            model = torch.compile(model)
            print("Compiled whole model successfully.")
        except Exception as exc:
            print(f"Warning: torch.compile failed for whole model: {exc}")

    macs = compute_macs(model, device)

    warmup_model(model, vid.shape[0], device, batch_size=args.batch_size)
    decode_fps = measure_decode_fps(model, vid.shape[0], device, batch_size=args.batch_size)
    psnr = measure_psnr(model, vid, device, batch_size=args.batch_size, debug_psnr=args.debug_psnr)
    encode_fps = measure_encode_fps(model, vid, device, batch_size=args.batch_size)

    print(f"Decode FPS: {decode_fps:.2f}")
    print(f"Encode FPS: {encode_fps:.2f}")
    print(f"MACs: {macs:.2f}")
    print(f"Clamped PSNR: {psnr:.4f}")


if __name__ == "__main__":
    main()
