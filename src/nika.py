"""Core Nika model definitions, training entrypoints, and visualization helpers."""

import os
import time

import torch
import torch.nn.functional as F

import argparse
import concurrent.futures
from torch.optim.lr_scheduler import LambdaLR

from soap import SOAP
from load_data import load_video_frames
from configs import REFERENCES
from nika_block import NikaBlock


def feature_test(vid, name, config, device):
    """Train the main Nika model on one video sequence.

    Args:
        vid: Video tensor shaped ``(T, C, H, W)`` containing training frames.
        name: Name of the sequence, used for checkpoint naming.
        config: Key into ``REFERENCES`` selecting the model hyperparameters.
        device: Device on which to run training.
    """
    batch_size = 16

    model_kwargs = REFERENCES[config]
    # Bunny-specific: double base feature-grid channels.
    if name == "bunny":
        model_kwargs = model_kwargs.copy()
        model_kwargs["base_grid_channels"] = int(model_kwargs["base_grid_channels"]) * 2
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        operator_steps=2,
        device=device,
    )

    model = torch.compile(model)
    base_lr = 3e-3
    opt = SOAP(
        model.parameters(),
        lr=base_lr,
        betas=(0.95, 0.95),
        weight_decay=3e-4,
        precondition_frequency=10,
        normalize_grads=False,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",          # because PSNR higher is better
        factor=0.5,          # halve LR
        patience=40,         # wait 30 eval epochs without real improvement
        threshold=0.015,      # require +0.015 dB improvement
        threshold_mode="abs",
        cooldown=25,         # wait after a drop before watching again
        min_lr=3e-4,
    )

    best_psnr = float('-inf')
    best_epoch = -1
    sequence_name = os.path.basename(str(name).rstrip("/"))

    # background executor for checkpoint saving so disk I/O doesn't stall training loop
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _save_checkpoint_in_bg(model_obj, path, epoch, psnr):
        # retained for backward-compat; not used below
        try:
            state = model_obj.state_dict()
        except Exception:
            state = None
        try:
            if state is not None:
                torch.save(state, path)
            else:
                torch.save({}, path)
            try:
                os.sync()
            except Exception:
                pass
            print(f"New best model saved in background at epoch {epoch} with PSNR: {psnr:.2f}")
        except Exception as e:
            print(f"Background save failed: {e}")

    def _save_state_cpu_in_bg(state_cpu, path, epoch, psnr):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_cpu, path)
            try:
                os.sync()
            except Exception:
                pass
            print(f"Background saved {path} (epoch {epoch}, PSNR: {psnr:.2f})")
        except Exception as e:
            print(f"Background save failed for {path}: {e}")

    use_cuda_timing = isinstance(device, str) and ("cuda" in device.lower())

    for epoch in range(3000):
        epoch_psnr_sum = 0.0
        if use_cuda_timing:
            epoch_starter = torch.cuda.Event(enable_timing=True)
            epoch_ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            epoch_starter.record()
        else:
            start_time = time.time()
        num_batches = (vid.shape[0] + batch_size - 1) // batch_size
        for t in range(num_batches):
            min_t = t * batch_size
            max_t = min((t + 1) * batch_size, vid.shape[0])
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.float() / (vid.shape[0] - 1)
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            prediction = model(norm_t_batch)
            mse = F.mse_loss(prediction, batch_gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            # PSNR is already computed from a batch-mean MSE, so extra batch-size scaling is unnecessary.
            frame_loss = -psnr
            frame_loss.backward()
            opt.step()
            epoch_psnr_sum += psnr.item()
        if scheduler is not None:
            scheduler.step(psnr)
        # epoch timing: total time, per-frame average, and equivalent FPS
        if use_cuda_timing:
            epoch_ender.record()
            torch.cuda.synchronize(device)
            epoch_time = epoch_starter.elapsed_time(epoch_ender) / 1000.0
        else:
            epoch_time = time.time() - start_time
        average_frame_time = epoch_time / float(vid.shape[0])
        fps = (float(vid.shape[0]) / epoch_time) if epoch_time > 0.0 else float('inf')
        epoch_psnr = epoch_psnr_sum / float(num_batches)
        print(
            f"Epoch {epoch} loss: {-epoch_psnr:.4f}, time: {average_frame_time:.5f}s/frame (FPS: {fps:.2f}), PSNR: {epoch_psnr:.2f}"
        )

        if epoch_psnr > best_psnr and (epoch - best_epoch >= 10 or best_epoch == -1):
            best_psnr = epoch_psnr
            best_epoch = epoch
            model_path = os.path.join("models", f"{config}-{sequence_name}-epoch{epoch}-psnr{best_psnr:.2f}.torch")
            # move state to CPU on main thread to avoid GPU-side sync inside background thread
            try:
                state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            except Exception:
                state_cpu = None
            # schedule background save (does not block main thread)
            if state_cpu is not None:
                executor.submit(_save_state_cpu_in_bg, state_cpu, model_path, epoch, float(best_psnr))
            else:
                executor.submit(_save_checkpoint_in_bg, model, model_path, epoch, float(best_psnr))

    print(f"Best PSNR achieved: {best_psnr:.2f} at epoch {best_epoch}")
    # wait for any outstanding background saves to finish before returning
    executor.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nika feature test with simple CLI args")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu' (e.g. 0 or cpu). Can also pass 'cuda:0' style string.")
    parser.add_argument("--name", default="bunny", help="Benchmark sequence name (folder under static/benchmarks)")
    parser.add_argument("--config", default="small", help="Model config key from REFERENCES (e.g. small)")
    args = parser.parse_args()

    # Resolve device string: allow numeric index, 'cpu', or full device string
    dev_arg = str(args.device)
    if dev_arg.lower() == "cpu":
        device = "cpu"
    elif dev_arg.isdigit():
        device = f"cuda:{dev_arg}"
    else:
        device = dev_arg

    name = args.name
    torch.manual_seed(42)
    vid = load_video_frames(f"static/benchmarks/{name}", device, dtype=torch.uint8, normalize=False)
    torch.set_float32_matmul_precision("high")
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    feature_test(vid, name, args.config, device=device)
