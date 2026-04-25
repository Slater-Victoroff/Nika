"""Load frame sequences from on-disk PNG folders into torch tensors."""

import glob
import torch
import imageio.v3 as iio


def load_video_frames(
    dir_path,
    device="cuda",
    dtype=torch.float32,
    normalize=True,

):
    """Load up to ``max_frames`` PNG frames from a directory into device memory.

    Args:
        dir_path: Directory containing numbered PNG frame files.
        device: Torch device identifier used for the returned tensor.
        dtype: Tensor dtype to apply to the loaded frames.
        normalize: Whether to scale image intensities into the ``[0, 1]`` range.

    Returns:
        A tensor shaped ``(T, C, H, W)`` containing the loaded video frames.
    """
    torch.cuda.set_device(device)
    paths = sorted(glob.glob(f"{dir_path}/*.png"))
    if not paths:
        raise RuntimeError(f"No frames found in {dir_path}")
    first = iio.imread(paths[0], plugin="pillow")
    H, W = first.shape[:2]

    # Preallocate CPU tensor (pinned for faster HtoD)
    vid_cpu = torch.empty((len(paths), 3, H, W), dtype=dtype, pin_memory=True)

    # Fill preallocated buffer
    for t, p in enumerate(paths):
        if t % 50 == 0:
            print(f"Loading frame {t}/{len(paths)}")
        if normalize:
            arr = iio.imread(p, plugin="pillow").astype("float32") / 255.0
        else:
            arr = iio.imread(p, plugin="pillow")
        arr = torch.from_numpy(arr)
        lin = arr.permute(2, 0, 1).contiguous()         # 3,H,W
        vid_cpu[t].copy_(lin)

    return vid_cpu.to(device=device, dtype=dtype, non_blocking=True)
