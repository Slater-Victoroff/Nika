"""Utility workflows for selecting checkpoints, benchmarking, and visualization."""

import os
import time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.autograd import grad
try:
    from torch.utils.flop_counter import FlopCounterMode
except Exception:
    FlopCounterMode = None

import re
import glob
from load_data import load_video_frames
from nika import NikaBlock
from soap import SOAP
from configs import REFERENCES
import subprocess
import copy

def get_best_model(model_dir, vid_shape, vid_name, config, device):
    """Load the highest-PSNR checkpoint that matches a video/config pair.

    Args:
        model_dir: Directory containing trained checkpoint files.
        vid_shape: Video tensor shape used to size the model.
        vid_name: Video identifier embedded in checkpoint filenames.
        config: Configuration name embedded in checkpoint filenames.
        device: Device on which to instantiate the model.

    Returns:
        A ``NikaBlock`` loaded from the best-scoring matching checkpoint.
    """
    all_models = glob.glob(f"{model_dir}/{config}-{vid_name}-*.torch")
    if not all_models:
        raise ValueError(f"No models found for {vid_name} with config {config}")

    # Sort models by PSNR (extracting the PSNR value from the filename)
    def extract_psnr(filename):
        """Parse the PSNR suffix from a checkpoint filename.

        Args:
            filename: Checkpoint filename containing a ``psnr`` token.

        Returns:
            The PSNR value encoded in the filename as a float.
        """
        match = re.search(r'psnr([0-9]+(?:\.[0-9]+)?)', filename)
        if match:
            return float(match.group(1))
        raise ValueError(f"Could not extract PSNR from filename: {filename}")
    all_models.sort(key=extract_psnr)

    print(f"Best model for {vid_name} with config {config}: {all_models[-1]}")
    model_kwargs = REFERENCES[config]
    if vid_name == "bunny":
        model_kwargs = model_kwargs.copy()
        model_kwargs["base_grid_channels"] = int(model_kwargs["base_grid_channels"]) * 2
    model = NikaBlock(
        target_shape=[4, vid_shape[2], vid_shape[3], vid_shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        operator_steps=2,
        device=device,
    )
    model_path = all_models[-1]
    # model_path = "models/ref_models/small-beauty-epoch1999-psnr33.36.torch"
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle torch.compile saved state_dicts with _orig_mod prefixes
        if any("_orig_mod" in k for k in state_dict.keys()):
            cleaned_state = {}
            for k, v in state_dict.items():
                cleaned_key = k
                if cleaned_key.startswith("_orig_mod."):
                    cleaned_key = cleaned_key[len("_orig_mod."):]
                cleaned_key = cleaned_key.replace("._orig_mod.", ".")
                cleaned_state[cleaned_key] = v
            model.load_state_dict(cleaned_state)
        else:
            raise e
    return model


def benchmark_psnr(basedir, vid_name, config, device):
    """Benchmark one trained checkpoint on a video sequence and save diagnostics.

    Args:
        basedir: Root directory containing the benchmark frame folders.
        vid_name: Name of the video sequence to evaluate.
        config: Model preset to load from ``models/ref_models``.
        device: Device on which to run inference.
    """
    vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    model = get_best_model(f"models/ref_models/", vid.shape, vid_name, config, device)

    core_image = model.grid_features.grid.data.cpu().numpy().copy()
    print(f"Core image shape: {core_image.shape}, value range: [{core_image.min()}, {core_image.max()}]")

    # Convert to tensor and make it a valid image: [C, H, W], normalized to [0, 1]
    core_t = torch.from_numpy(core_image).squeeze(-1).to(torch.float32)
    # Use first 3 channels for RGB visualization
    core_t = core_t[:3, ...]
    # Percentile-based normalization to avoid gray-looking images
    q_low = torch.quantile(core_t, 0.01)
    q_high = torch.quantile(core_t, 0.99)
    core_t = (core_t - q_low) / (q_high - q_low + 1e-8)
    core_t = core_t.clamp(0.0, 1.0)
    save_image(core_t, f"visuals/{vid_name}/{config}/core_image.png")
    os.makedirs(f"visuals/{vid_name}/{config}/preds", exist_ok=True)
    os.makedirs(f"visuals/{vid_name}/{config}/residual", exist_ok=True)
    model.eval()
    total_psnr = 0.0
    num_frames = vid.shape[0]

    batch_size = 32
    num_batches = (num_frames + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in range(num_batches):
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.to(torch.float32) / max(1, (num_frames - 1))
            prediction = model(norm_t_batch)
            residual = prediction - batch_gt
            residual_max = residual.max(); residual_min = residual.min()
            for i in range(prediction.shape[0]):
                save_image(prediction[i], f"visuals/{vid_name}/{config}/preds/pred_frame{min_t + i:03d}.png")
                # Map residual to [0, 1] by normalizing to its min/max per-frame
                res = residual[i]
                res_norm = torch.abs(res) * 5.0  # scale up for visibility
                save_image(res_norm, f"visuals/{vid_name}/{config}/residual/residual_frame{min_t + i:03d}.png")
                # FFT of residual: log-magnitude, centered, normalized per-frame
                try:
                    fft_res = torch.fft.fft2(res)
                    fft_mag = torch.abs(torch.fft.fftshift(fft_res, dim=(-2, -1)))
                    fft_log = torch.log1p(fft_mag)
                    fft_norm = fft_log / (fft_log.max() + 1e-8)
                    save_image(fft_norm, f"visuals/{vid_name}/{config}/residual_fft/residual_fft_frame{min_t + i:03d}.png")
                except Exception:
                    # If FFT/save fails, skip silently to avoid breaking benchmark
                    pass
                mse = F.mse_loss(prediction[i].clamp(0, 1), batch_gt[i])
                psnr = 10 * torch.log10(1 / (mse + 1e-8))
                total_psnr += psnr.item()
                if (min_t + i) % 100 == 0:
                    print(f"Processed frame {min_t + i}, PSNR: {psnr:.4f}")

    avg_psnr = total_psnr / num_frames
    print(f"Average PSNR: {avg_psnr:.4f}")

    # Timing run
    if "cuda" in device:
        torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.to(torch.float32) / max(1, (num_frames - 1))
            _ = model(norm_t_batch)
    if "cuda" in device:
        torch.cuda.synchronize(device)
    end_time = time.time()
    print(f"Timing run took {end_time - start_time:.4f} seconds")


def _measure_macs(eval_model, norm_t_all, batch_size, num_frames, device):
    """Estimate MACs for a single forward pass (FLOPs / 2).

    Uses an uncompiled model to avoid torch.compile hook warnings inside
    FlopCounterMode.

    Returns:
        A formatted string like ``"4.72e+10"`` or ``"MACs unavailable"``.
    """
    if FlopCounterMode is None:
        return "MACs unavailable"
    sample_idx = torch.randint(0, num_frames, (batch_size,), device=device)
    sample_norm_t = (sample_idx.float() / max(1, num_frames - 1)).requires_grad_(True)
    with torch.enable_grad(), FlopCounterMode(display=False) as fcm:
        _ = eval_model(sample_norm_t)
    try:
        total_flops = fcm.get_total_flops()
    except Exception:
        total_flops = None
    if total_flops is None:
        return "MACs unavailable"
    return f"{total_flops / 2.0:.3e}"


def _measure_psnr(eval_model, vid, norm_t_all, batch_size, num_frames):
    """Compute mean per-frame PSNR over the full sequence.

    Args:
        eval_model: Model in eval mode (uncompiled is fine).
        vid: ``uint8`` video tensor ``[T, C, H, W]`` on device.
        norm_t_all: Normalised time values ``[T]`` in ``[0, 1]``.
        batch_size: Frames per inference batch.
        num_frames: Total number of frames.

    Returns:
        Mean per-frame PSNR as a float.
    """
    total_psnr = 0.0
    total_count = 0
    num_batches = (num_frames + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in range(num_batches):
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            pred = eval_model(norm_t_all[min_t:max_t]).clamp(0, 1)
            frame_mse = (pred - batch_gt).pow(2).view(pred.shape[0], -1).mean(dim=1)
            frame_psnr = 10.0 * torch.log10(1.0 / (frame_mse + 1e-8))
            total_psnr += frame_psnr.sum().item()
            total_count += frame_psnr.numel()
    return total_psnr / max(1, total_count)


def _profile_decode(eval_model, norm_t_all, num_frames, runs, trace_dir):
    """Capture a Perfetto decode trace (uncompiled model, frame-by-frame).

    Args:
        eval_model: Model in eval mode. Should be *uncompiled* so the profiler
            captures op-level breakdown rather than fused kernels.
        norm_t_all: Normalised time values ``[T]`` in ``[0, 1]``.
        num_frames: Total number of frames.
        runs: Number of active profiler steps to record.
        trace_dir: Directory where the trace JSON will be written.
    """
    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=0, active=runs, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(1 + runs):
            for i in range(num_frames):
                _ = eval_model(norm_t_all[i:i + 1])
            prof.step()


def _time_decode(eval_model, norm_t_all, num_frames, warmup_iters, repeats, discard_first, device):
    """Time the decode (inference) loop over the full sequence.

    Args:
        eval_model: Compiled model in eval mode.
        norm_t_all: Normalised time values ``[T]`` in ``[0, 1]``.
        num_frames: Total number of frames.
        warmup_iters: Number of un-timed warm-up passes.
        repeats: Number of timed passes to average.
        discard_first: Whether to discard the first timed run.
        device: CUDA device string.

    Returns:
        ``(fps, avg_ms_per_run)`` tuple.
    """
    with torch.no_grad():
        for _ in range(warmup_iters):
            for i in range(num_frames):
                _ = eval_model(norm_t_all[i:i + 1])

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    total_frames = 0
    timed_runs = repeats + (1 if discard_first else 0)
    with torch.no_grad():
        for run_idx in range(timed_runs):
            torch.cuda.synchronize(device)
            starter.record()
            for i in range(num_frames):
                _ = eval_model(norm_t_all[i:i + 1])
            ender.record()
            torch.cuda.synchronize(device)
            if discard_first and run_idx == 0:
                continue
            total_ms += starter.elapsed_time(ender)
            total_frames += num_frames

    measured = max(1, timed_runs - (1 if discard_first else 0))
    avg_ms = total_ms / measured
    fps = (total_frames / measured) / (avg_ms / 1000.0)
    return fps, avg_ms


def _run_train_epoch(train_model, train_opt, vid_cpu, norm_t_all, num_frames, batch_size, device):
    """One full training epoch: forward + PSNR loss + backward + optimizer step.

    Batches are transferred from pinned CPU memory non-blocking so the H2D DMA
    can overlap with GPU compute of the previous batch.

    Args:
        train_model: Model in train mode.
        train_opt: Optimizer instance.
        vid_cpu: Float32 GT frames ``[T, C, H, W]`` in pinned CPU memory.
        norm_t_all: Normalised time values ``[T]`` on device.
        num_frames: Total number of frames.
        batch_size: Frames per mini-batch.
        device: CUDA device string.

    Returns:
        Accumulated scalar loss for the epoch.
    """
    num_batches = (num_frames + batch_size - 1) // batch_size
    train_opt.zero_grad(set_to_none=True)
    epoch_loss = 0.0
    for batch_idx in range(num_batches):
        min_t = batch_idx * batch_size
        max_t = min((batch_idx + 1) * batch_size, num_frames)
        batch_gt = vid_cpu[min_t:max_t].to(device, non_blocking=True)
        batch_norm_t = norm_t_all[min_t:max_t]
        prediction = train_model(batch_norm_t)
        mse = F.mse_loss(prediction, batch_gt)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        batch_loss = (-psnr).mean() / num_batches
        batch_loss.backward()
        epoch_loss += batch_loss.item()
    train_opt.step()
    return epoch_loss


def _profile_encode(base_model, vid_cpu, norm_t_all, num_frames, batch_size, runs, trace_dir, device):
    """Capture a Perfetto encode trace using an isolated copy of the model.

    A fresh deepcopy and optimizer are used so profiling doesn't affect the
    weights or optimizer state of the model used for timed benchmarking.

    Args:
        base_model: Uncompiled source model (will be deepcopied).
        vid_cpu: Float32 GT frames in pinned CPU memory.
        norm_t_all: Normalised time values ``[T]`` on device.
        num_frames: Total number of frames.
        batch_size: Frames per mini-batch.
        runs: Number of active profiler steps to record.
        trace_dir: Directory where the trace JSON will be written.
        device: CUDA device string.
    """
    profile_model = copy.deepcopy(base_model)
    profile_model.train()
    profile_opt = SOAP(profile_model.parameters(), lr=1e-2, weight_decay=0)
    with torch.enable_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=0, active=runs, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(1 + runs):
            _run_train_epoch(profile_model, profile_opt, vid_cpu, norm_t_all, num_frames, batch_size, device)
            prof.step()


def _time_encode(encoding_model, opt, vid_cpu, norm_t_all, num_frames, batch_size, warmup_iters, repeats, discard_first, device):
    """Time the encode (training epoch) loop.

    Args:
        encoding_model: Compiled model in train mode.
        opt: Optimizer instance (state is advanced during timing).
        vid_cpu: Float32 GT frames in pinned CPU memory.
        norm_t_all: Normalised time values ``[T]`` on device.
        num_frames: Total number of frames.
        batch_size: Frames per mini-batch.
        warmup_iters: Number of un-timed warm-up epochs.
        repeats: Number of timed epochs to average.
        discard_first: Whether to discard the first timed run.
        device: CUDA device string.

    Returns:
        ``(fps, avg_ms_per_epoch)`` tuple.
    """
    with torch.enable_grad():
        for _ in range(warmup_iters):
            _run_train_epoch(encoding_model, opt, vid_cpu, norm_t_all, num_frames, batch_size, device)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    total_epochs = 0
    timed_runs = repeats + (1 if discard_first else 0)
    with torch.enable_grad():
        for run_idx in range(timed_runs):
            torch.cuda.synchronize(device)
            starter.record()
            _run_train_epoch(encoding_model, opt, vid_cpu, norm_t_all, num_frames, batch_size, device)
            ender.record()
            torch.cuda.synchronize(device)
            if discard_first and run_idx == 0:
                continue
            total_ms += starter.elapsed_time(ender)
            total_epochs += 1

    avg_ms = total_ms / max(1, total_epochs)
    fps = (num_frames * total_epochs) / (total_ms / 1000.0)
    return fps, avg_ms


def benchmark_encoding_decoding_and_macs(
    basedir,
    vid_name,
    config,
    device,
    batch_size=2,
    warmup_iters=5,
    repeats=10,
    profile_decode_runs=4,
    profile_encode_runs=4,
    discard_first_timed_run=True,
):
    """Full encode/decode benchmark: PSNR, MACs, decode speed, encode speed, profiles.

    Args:
        basedir: Root directory containing benchmark frame folders.
        vid_name: Video sequence name.
        config: Model preset key from ``REFERENCES``.
        device: CUDA device string (e.g. ``"cuda:0"``).
        batch_size: Frames per batch for PSNR eval and encode timing.
        warmup_iters: Un-timed warm-up iterations before timing.
        repeats: Number of timed iterations to average.
        profile_decode_runs: Profiler steps to capture for the decode trace.
        profile_encode_runs: Profiler steps to capture for the encode trace.
        discard_first_timed_run: Drop the first timed run to avoid cold-start bias.
    """
    if "cuda" not in device:
        raise ValueError("This benchmark uses CUDA events; please use a CUDA device.")

    vid = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    model = get_best_model("models/ref_models/", vid.shape, vid_name, config, device)

    num_frames = int(vid.shape[0])
    norm_t_all = torch.linspace(0.0, 1.0, steps=num_frames, device=device, dtype=torch.float32)

    # Separate eval and encode model instances so train/eval modes don't interfere.
    eval_model = copy.deepcopy(model)
    eval_model.eval()

    encoding_model = copy.deepcopy(model)
    encoding_model.train()
    encoding_model = torch.compile(encoding_model, mode="default")
    opt = SOAP(encoding_model.parameters(), lr=1e-2, weight_decay=0)

    # GT frames in pinned CPU memory for non-blocking H2D transfers during encode.
    vid_cpu = vid.cpu().to(torch.float32).div_(255.0).pin_memory()

    # --- MACs ---
    macs_str = _measure_macs(eval_model, norm_t_all, batch_size, num_frames, device)

    # --- PSNR ---
    avg_frame_psnr = _measure_psnr(eval_model, vid, norm_t_all, batch_size, num_frames)

    # --- Profiles (uncompiled models for op-level trace detail) ---
    trace_root = os.path.join("profiles", "encoding_decoding_benchmark")
    decode_trace_dir = os.path.join(trace_root, "decode")
    encode_trace_dir = os.path.join(trace_root, "encode")
    os.makedirs(decode_trace_dir, exist_ok=True)
    os.makedirs(encode_trace_dir, exist_ok=True)

    if profile_decode_runs > 0:
        _profile_decode(eval_model, norm_t_all, num_frames, profile_decode_runs, decode_trace_dir)

    if profile_encode_runs > 0:
        _profile_encode(model, vid_cpu, norm_t_all, num_frames, batch_size, profile_encode_runs, encode_trace_dir, device)

    # Compile eval model for timing (after profiling to avoid hook warnings).
    eval_model = torch.compile(eval_model, mode="reduce-overhead")

    # --- Timing ---
    decode_fps, decode_avg_ms = _time_decode(
        eval_model, norm_t_all, num_frames, warmup_iters, repeats, discard_first_timed_run, device
    )
    encode_fps, encode_avg_ms = _time_encode(
        encoding_model, opt, vid_cpu, norm_t_all, num_frames, batch_size,
        warmup_iters, repeats, discard_first_timed_run, device,
    )

    print(
        " | ".join([
            f"Encoding speed: {encode_fps:.2f} fps ({encode_avg_ms:.2f} ms/epoch)",
            f"Decoding speed: {decode_fps:.2f} fps ({decode_avg_ms:.2f} ms/{num_frames} frames)",
            f"MACs per forward: {macs_str}",
            f"PSNR (avg frame): {avg_frame_psnr:.2f} dB",
            f"Decode profile: {decode_trace_dir}",
            f"Encode profile: {encode_trace_dir}",
        ])
    )


def make_mp4(png_frame_dir, output_path="output.mp4", base_name="pred_frame", fps=24):
    """Assemble numbered PNG frames into an MP4 using ffmpeg.

    Args:
        png_frame_dir: Directory containing sequentially numbered frame images.
        output_path: Destination MP4 file path.
        base_name: Filename prefix used before the numeric frame index.
        fps: Output frame rate for the encoded video.
    """
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


def module_visualization(basedir, vid_name, n_frames, config, device, variants=None, batch_size=1, commit=None):
    """Render module-ablation visualizations for a trained checkpoint.

    Args:
        basedir: Root directory containing benchmark frame folders.
        vid_name: Video identifier to visualize.
        n_frames: Number of frames to render for the visualization run.
        config: Model preset whose checkpoint should be loaded.
        device: Device on which to run inference.
        variants: Optional list of visualization variants to export.
        batch_size: Number of frames to render per inference batch.
        commit: Optional flag to further specify target model
    """
    if variants is None:
        variants = ['baseline', 'only_real_grid', 'only_realt', 'only_complex_grid', 'only_complext', 'temporal_operators']

    # Determine model input shape using a single-frame probe (avoids loading full video)
    probe = load_video_frames(f"{basedir}/{vid_name}", device, dtype=torch.uint8, normalize=False)
    probe_shape = probe.shape  # (T_probe, C, H, W)
    # Use the provided `n_frames` for temporal length, but match spatial/channel dims from probe
    vid_shape = [n_frames, probe_shape[1], probe_shape[2], probe_shape[3]]
    model = get_best_model(f"models/ref_models/", vid_shape, vid_name, config, device)
    # dir_suff = f"{commit}/{config}" if commit is not None else f"{config}"
    # model = get_best_model(f"models/ref_models/{dir_suff}", vid_shape, vid_name, config, device)
    model.eval()

    num_frames = int(n_frames)
    num_batches = (num_frames + batch_size - 1) // batch_size

    for v in variants:
        # create main preds dir for variant
        os.makedirs(f"visuals/{vid_name}/{config}/{v}/preds", exist_ok=True)
        # if storing forward/back separately, create subfolders
        if v == 'temporal_operators':
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/minus_one/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/plus_one/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/minus_two/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/plus_two/preds", exist_ok=True)
            os.makedirs(f"visuals/{vid_name}/{config}/{v}/full_operator_residual/preds", exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            print(f"Processing batch {batch_idx + 1}/{num_batches} for variants: {variants}")
            min_t = batch_idx * batch_size
            max_t = min((batch_idx + 1) * batch_size, num_frames)
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.float() / (num_frames - 1)  # normalize time for operators that use it

            # baseline call
            if 'baseline' in variants:
                out_base = model(norm_t_batch)

            # zeroed variants use forward flags
            if 'only_real_grid' in variants:
                out_real_grid = model(norm_t_batch, zero_real_tucker=True, zero_complex_tucker=True, zero_complex_grid=True)
            if 'only_realt' in variants:
                out_realt = model(norm_t_batch, zero_feature_grid=True, zero_complex_tucker=True, zero_complex_grid=True)
            if 'only_complex_grid' in variants:
                out_complex_grid = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_tucker=True)
            if 'only_complext' in variants:
                out_complext = model(norm_t_batch, zero_real_tucker=True, zero_feature_grid=True, zero_complex_grid=True)

            # forward/backward operators passed through upres
            if 'temporal_operators' in variants:
                # model.forward(..., return_operators=True) -> (refined, refined_forward, refined_backward)
                _, full_operator_residual = model(norm_t_batch, return_operators=True)
                # full_operator_residual = minus_one + plus_one + minus_two + plus_two

            # save per-variant frames
            for i in range(t_batch.shape[0]):
                idx = min_t + i
                if 'baseline' in variants:
                    save_image(out_base[i], f"visuals/{vid_name}/{config}/baseline/preds/pred_frame{idx:03d}.png")
                if 'only_real_grid' in variants:
                    save_image(out_real_grid[i], f"visuals/{vid_name}/{config}/only_real_grid/preds/pred_frame{idx:03d}.png")
                if 'only_realt' in variants:
                    save_image(out_realt[i], f"visuals/{vid_name}/{config}/only_realt/preds/pred_frame{idx:03d}.png")
                if 'only_complex_grid' in variants:
                    save_image(out_complex_grid[i], f"visuals/{vid_name}/{config}/only_complex_grid/preds/pred_frame{idx:03d}.png")
                if 'only_complext' in variants:
                    save_image(out_complext[i], f"visuals/{vid_name}/{config}/only_complext/preds/pred_frame{idx:03d}.png")
                if 'temporal_operators' in variants:
                    # save_image(minus_one[i], f"visuals/{vid_name}/{config}/temporal_operators/minus_one/preds/pred_frame{idx:03d}.png")
                    # save_image(plus_one[i], f"visuals/{vid_name}/{config}/temporal_operators/plus_one/preds/pred_frame{idx:03d}.png")
                    # save_image(minus_two[i], f"visuals/{vid_name}/{config}/temporal_operators/minus_two/preds/pred_frame{idx:03d}.png")
                    # save_image(plus_two[i], f"visuals/{vid_name}/{config}/temporal_operators/plus_two/preds/pred_frame{idx:03d}.png")
                    save_image(full_operator_residual[i], f"visuals/{vid_name}/{config}/temporal_operators/full_operator_residual/preds/pred_frame{idx:03d}.png")

    # make mp4s for each variant
    for v in variants:
        try:
            if v == 'temporal_operators':
                for op in ['minus_one', 'plus_one', 'minus_two', 'plus_two', 'full_operator_residual']:
                    src_dir = f"visuals/{vid_name}/{config}/{v}/{op}/preds"
                    out_path = f"visuals/{vid_name}/{config}/{v}/{op}.mp4"
                    make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)

            else:
                src_dir = f"visuals/{vid_name}/{config}/{v}/preds"
                out_path = f"visuals/{vid_name}/{config}/{v}.mp4"
                make_mp4(src_dir, output_path=out_path, base_name="pred_frame", fps=24)
        except Exception as e:
            print(f"Failed to create mp4 for {v}: {e}")


if __name__ == "__main__":
    device = "cuda:0"
    name = "bunny"
    n_frames = 132
    config = "small"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    module_visualization("static/benchmarks", name, n_frames=n_frames, config=config, device=device)
    # benchmark_encoding_decoding_and_macs("static/benchmarks/uvg", name, config, device, batch_size=16)
    # benchmark_psnr("static/benchmarks/uvg", name, config, device)
    # make_mp4(f"visuals/{name}/{config}/preds", output_path=f"visuals/{name}/{config}/preds/output.mp4", base_name="pred_frame", fps=24)
    # make_mp4(f"visuals/{name}/{config}/residual", output_path=f"visuals/{name}/{config}/residual/output.mp4", base_name="residual_frame", fps=24)
