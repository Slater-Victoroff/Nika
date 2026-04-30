import os
import time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.autograd import grad

import re
import glob
from load_data import load_video_frames
from nika import NikaBlock
from soap import SOAP
from configs import REFERENCES
import subprocess
import copy
import argparse
import torch.profiler

# Optional: try to import a flop counting tool (fvcore). If unavailable, MACs reporting will be disabled.
try:
    from fvcore.nn.flop_count import FlopCountAnalysis as FlopCounterMode
except Exception:
    FlopCounterMode = None

def get_best_model(model_dir, vid_shape, vid_name, config, device):
    # allow passing paths like 'uvg/beauty' by using basename for model lookup
    model_vid = os.path.basename(vid_name)
    all_models = glob.glob(f"{model_dir}/{config}-{model_vid}-*.torch")
    if not all_models:
        raise ValueError(f"No models found for {vid_name} with config {config}")

    # Sort models by PSNR (extracting the PSNR value from the filename)
    def extract_psnr(filename):
        match = re.search(r'psnr([0-9]+(?:\.[0-9]+)?)', filename)
        if match:
            return float(match.group(1))
        raise ValueError(f"Could not extract PSNR from filename: {filename}")
    all_models.sort(key=extract_psnr)

    model_path = all_models[-1]
    print(f"Best model for {vid_name} with config {config}: {model_path}")

    # Build model kwargs from REFERENCES and apply dataset-specific overrides
    model_kwargs = dict(REFERENCES[config])
    # replicate `feature_test` behavior: double grid_ranks for 'bunny'
    if model_vid == 'bunny' and 'grid_ranks' in model_kwargs:
        model_kwargs['grid_ranks'] = model_kwargs['grid_ranks'] * 2

    model = NikaBlock(
        target_shape=[4, vid_shape[2], vid_shape[3], vid_shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        device=device,
    )
    state_dict = torch.load(model_path, map_location=device)
    # Normalize keys by removing `._orig_mod` for matching purposes, then
    # align checkpoint values to the model's expected keys. This handles
    # checkpoints saved both from compiled and uncompiled modules.
    ckpt_map = {}
    for k, v in state_dict.items():
        norm = k.replace('._orig_mod', '')
        ckpt_map[norm] = v

    mapped_sd = {}
    used = set()
    model_keys = list(model.state_dict().keys())
    for mk in model_keys:
        norm_mk = mk.replace('._orig_mod', '')
        if norm_mk in ckpt_map:
            mapped_sd[mk] = ckpt_map[norm_mk]
            used.add(norm_mk)

    # Load with strict=False to allow partial matches; report diffs for debugging.
    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    if missing:
        print(f"Warning: missing keys when loading {model_path}: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading {model_path}: {unexpected}")
    return model


def benchmark_psnr(basedir, vid_name, config, device):
    # Very small, explicit benchmark: evaluate each frame one-by-one.
    vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    model = get_best_model("models/ref_models", vid.shape, vid_name, config, device)

    model.eval()
    num_frames = int(vid.shape[0])
    total_psnr = 0.0

    # Simple per-frame evaluation (no saving, no batching, straightforward math)
    with torch.no_grad():
        for t in range(num_frames):
            gt = vid[t].to(torch.float32) / 255.0
            norm_t = torch.tensor([t / max(num_frames - 1, 1)], device=device, dtype=torch.float32)
            pred = model(norm_t)
            # model returns [1, C, H, W]
            pred_img = pred[0]
            # Clamp predictions to valid image range before comparison
            pred_img_clamped = pred_img.clamp(0.0, 1.0)
            mse = F.mse_loss(pred_img_clamped, gt)
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item()
            if (t % 100) == 0:
                print(f"Processed frame {t}, PSNR: {psnr:.4f}")

    avg_psnr = total_psnr / num_frames
    print(f"Average PSNR: {avg_psnr:.4f}")

    # Timing run: measure decode throughput using the same simple per-frame loop
    if "cuda" in device:
        torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        for t in range(num_frames):
            norm_t = torch.tensor([t / max(num_frames - 1, 1)], device=device, dtype=torch.float32)
            _ = model(norm_t)
    if "cuda" in device:
        torch.cuda.synchronize(device)
    end_time = time.time()
    elapsed = end_time - start_time
    fps = float(num_frames) / elapsed if elapsed > 0 else float('inf')
    print(f"Timing run took {elapsed:.4f} seconds — {fps:.2f} FPS (decode)")


def make_mp4(png_frame_dir, output_path="output.mp4", base_name="pred_frame", fps=24):
    # Assumes frames are named in order: frame000.png, frame001.png, ...
    input_pattern = os.path.join(png_frame_dir, f"{base_name}%03d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True)


def run_benchmark(basedir, vid_name, config, device, mode='psnr', variants=None, batch_size=10):
    """Dispatch helper to run common benchmark modes.

    mode: 'psnr' -> runs `benchmark_psnr`
          'ablation' -> runs `ablation_harness`
          'mp4' -> assembles an mp4 from `visuals/{vid_name}/{config}/preds`
    """
    mode = mode.lower()
    if mode == 'psnr':
        benchmark_psnr(basedir, vid_name, config, device)
    elif mode == 'ablation':
        ablation_harness(basedir, vid_name, config, device, variants=variants, batch_size=batch_size)
    elif mode == 'mp4':
        png_dir = f"visuals/{vid_name}/{config}/preds"
        if not os.path.exists(png_dir):
            raise ValueError(f"PNG frame dir not found: {png_dir}")
        make_mp4(png_dir, output_path=f"visuals/{vid_name}/{config}/preds/output.mp4", base_name="pred_frame", fps=24)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def profile_decode(basedir, vid_name, config, device, model_dir="models/ref_models", n_frames=20, batch_size=5, out_dir=None):
    # Disable torch.compile during profiling for more accurate traces
    os.environ['NIKA_USE_TORCH_COMPILE'] = '0'
    vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    model = get_best_model(model_dir, vid.shape, vid_name, config, device)
    model.eval()
    if out_dir is None:
        base = os.path.basename(vid_name)
        out_dir = os.path.join("profiles", "decoding")
        os.makedirs(out_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if 'cuda' in device:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            num_batches = (n_frames + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                min_t = batch_idx * batch_size
                max_t = min((batch_idx + 1) * batch_size, n_frames)
                t_idx = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
                norm_t = t_idx.float() / max(n_frames - 1, 1)
                _ = model(norm_t)

    out_name = f"{config}_{os.path.basename(vid_name)}_decode.json"
    out_path = os.path.join(out_dir, out_name)
    try:
        prof.export_chrome_trace(out_path)
        print(f"Saved decode profile trace to {out_path}")
    except Exception as e:
        print(f"Failed to export decode trace: {e}")

    try:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    except Exception:
        pass


def profile_encode(basedir, vid_name, config, device, model_dir="models/ref_models", n_frames=20, batch_size=5, steps=5, out_dir=None):
    # Disable torch.compile during profiling for more accurate traces
    os.environ['NIKA_USE_TORCH_COMPILE'] = '0'
    vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    model = get_best_model(model_dir, vid.shape, vid_name, config, device)
    model.train()
    if out_dir is None:
        out_dir = os.path.join("profiles", "encoding")
        os.makedirs(out_dir, exist_ok=True)

    # Small encoding workload: a few gradient steps on a single batch
    min_t = 0
    max_t = min(batch_size, n_frames)
    t_idx = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
    norm_t = t_idx.float() / max(n_frames - 1, 1)
    target = vid[min_t:max_t].to(torch.float32) / 255.0

    opt = SOAP(list(model.parameters()), lr=1e-2)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if 'cuda' in device:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            pred = model(norm_t)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

    out_name = f"{config}_{os.path.basename(vid_name)}_encode.json"
    out_path = os.path.join(out_dir, out_name)
    try:
        prof.export_chrome_trace(out_path)
        print(f"Saved encode profile trace to {out_path}")
    except Exception as e:
        print(f"Failed to export encode trace: {e}")

    try:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    except Exception:
        pass


def ablation_harness(basedir, vid_name, config, device, variants=None, batch_size=10):
    """Run several ablation variants using `NikaBlock.forward` flags and save frames.

    Variants (default):
      - 'baseline'
      - 'no_tucker' (zero both real and complex tucker heads)
      - 'zero_real'
      - 'zero_complex'
      - 'forward_backward_upres' (use the forward/back outputs passed through `upres`)
    """
    if variants is None:
        variants = ['baseline', 'only_grid', 'only_realt', 'only_complext', 'gridless', 'forward_backward_upres']

    # Determine model input shape using a single-frame probe (avoids loading full video)
    probe = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    probe_shape = probe.shape  # (T_probe, C, H, W)

    vid_shape = [probe_shape[0], probe_shape[1], probe_shape[2], probe_shape[3]]
    model = get_best_model("models/ref_models", vid_shape, vid_name, config, device)
    model.eval()

    num_frames = int(probe_shape[0])
    num_batches = (num_frames + batch_size - 1) // batch_size

    for v in variants:
        # create main preds dir for variant
        os.makedirs(f"visuals/{vid_name}/{config}/{v}/preds", exist_ok=True)
        # if storing forward/back separately, create subfolders
        if v == 'forward_backward_upres':
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/forward/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/backward/preds", exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            t_idx = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t = t_idx.float() / max(num_frames - 1, 1)

            # baseline call
            if 'baseline' in variants:
                out_base = model(norm_t)

            # zeroed variants use forward flags
            if 'only_grid' in variants:
                out_no_tucker = model(norm_t, zero_real_tucker=True, zero_complex_tucker=True)
            if 'only_realt' in variants:
                out_zero_real = model(norm_t, zero_complex_tucker=True, zero_feature_grid=True)
            if 'only_complext' in variants:
                out_zero_complex = model(norm_t, zero_real_tucker=True, zero_feature_grid=True)
            if 'gridless' in variants:
                out_backless = model(norm_t, zero_feature_grid=True)

            # forward/backward operators passed through upres
            if 'forward_backward_upres' in variants:
                # model.forward(..., return_operators=True) -> (refined, refined_forward, refined_backward)
                _, refined_forward, refined_backward = model(norm_t, return_operators=True)
                out_forward = refined_forward
                out_backward = refined_backward

            # save per-variant frames
            for i in range(norm_t.shape[0]):
                idx = min_t + i
                if 'baseline' in variants:
                    save_image(out_base[i], f"visuals/{vid_name}/{config}/baseline/preds/pred_frame{idx:03d}.png")
                if 'only_grid' in variants:
                    save_image(out_no_tucker[i], f"visuals/{vid_name}/{config}/only_grid/preds/pred_frame{idx:03d}.png")
                if 'only_realt' in variants:
                    save_image(out_zero_real[i], f"visuals/{vid_name}/{config}/only_realt/preds/pred_frame{idx:03d}.png")
                if 'only_complext' in variants:
                    save_image(out_zero_complex[i], f"visuals/{vid_name}/{config}/only_complext/preds/pred_frame{idx:03d}.png")
                if 'gridless' in variants:
                    save_image(out_backless[i], f"visuals/{vid_name}/{config}/gridless/preds/pred_frame{idx:03d}.png")
                if 'forward_backward_upres' in variants:
                    save_image(out_forward[i], f"visuals/{vid_name}/{config}/forward_backward_upres/forward/preds/pred_frame{idx:03d}.png")
                    save_image(out_backward[i], f"visuals/{vid_name}/{config}/forward_backward_upres/backward/preds/pred_frame{idx:03d}.png")

    # make mp4s
    for v in variants:
        try:
            if v == 'forward_backward_upres':
                fwd_src = f"visuals/{vid_name}/{config}/{v}/forward/preds"
                bwd_src = f"visuals/{vid_name}/{config}/{v}/backward/preds"
                make_mp4(fwd_src, output_path=f"visuals/{vid_name}/{config}/forward.mp4", base_name="pred_frame", fps=24)
                make_mp4(bwd_src, output_path=f"visuals/{vid_name}/{config}/backward.mp4", base_name="pred_frame", fps=24)
            else:
                src_dir = f"visuals/{vid_name}/{config}/{v}/preds"
                out_path = f"visuals/{vid_name}/{config}/{v}.mp4"
                make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)
        except Exception as e:
            print(f"Failed to create mp4 for {v}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nika benchmarks and profiling")
    parser.add_argument("--basedir", default="static/benchmarks", help="Base video directory")
    parser.add_argument("--name", required=True, help="Video name (e.g., bunny, bosphorus)")
    parser.add_argument("--config", default="small", help="Config name from configs.REFERENCES")
    parser.add_argument("--device", default="cuda:0", help="Device to run on, e.g. cuda:0 or cpu")
    parser.add_argument("--mode", default="ablation", choices=["psnr", "ablation", "mp4", "profile"], help="Mode to run")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--profile_type", default="both", choices=["encode", "decode", "both"], help="Which profile traces to capture when mode=profile")
    parser.add_argument("--out_dir", default=None, help="Output directory for profiles (overrides default)")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Normalize device string: allow passing `1` or `0` and convert to `cuda:1`.
    device = args.device
    if isinstance(device, str):
        # purely numeric -> cuda:N
        if re.fullmatch(r"\d+", device):
            device = f"cuda:{device}"
        # common shorthand like 'cuda0' -> 'cuda:0'
        elif re.fullmatch(r"cuda\d+", device):
            device = device.replace('cuda', 'cuda:')

    if args.mode == 'profile':
        if args.profile_type in ('decode', 'both'):
            profile_decode(args.basedir, args.name, args.config, device, batch_size=args.batch_size, out_dir=args.out_dir)
        if args.profile_type in ('encode', 'both'):
            profile_encode(args.basedir, args.name, args.config, device, batch_size=args.batch_size, out_dir=args.out_dir)
    else:
        run_benchmark(args.basedir, args.name, args.config, device, mode=args.mode, variants=None, batch_size=args.batch_size)
