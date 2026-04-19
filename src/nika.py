"""Core Nika model definitions, training entrypoints, and visualization helpers."""

import os
import time
import math
import glob
from functools import partial

import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F

import numpy as np
import imageio.v3 as iio
from PIL import Image

from soap import SOAP
import random
from load_data import load_video_frames
from encoding_utils import FourierEncoding
from configs import REFERENCES


class SimpleTuckerFactor(nn.Module):
    def __init__(self, target_dim, rank, is_complex=False, base_mag=1e-2, device='cuda'):
        """
        Initialize a factor matrix, chunking large dimensions for PyTorch stability.

        Args:
            target_dim: Output dimension of the factor matrix.
            rank: Tucker rank represented by the factor.
            is_complex: Whether to represent the factor with real and imaginary parts.
            base_mag: Standard deviation scale used for random initialization.
            device: Device on which to allocate the factor parameters.
        """
        super().__init__()
        self.target_dim = target_dim
        self.is_complex = is_complex
        if self.is_complex:
            self.U_real = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)
            self.U_imag = nn.Parameter(torch.zeros(target_dim, rank, device=device))
        else:
            self.U = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)
        self.dU = 1 / target_dim
        self.device = device
    
    def forward(self, target=None, index=None):
        if target is not None and index is not None:
            raise ValueError("Cannot specify both target and index")
        if target is None and index is None:
            if self.is_complex:
                return torch.complex(self.U_real, self.U_imag)
            return self.U
        if target is not None:
            target_t = torch.as_tensor(target, device=self.device, dtype=torch.float32)
            target_index = torch.floor(target_t * float(self.target_dim)).to(torch.long)
            index = torch.clamp(target_index, max=self.target_dim - 1, min=0)
        if self.is_complex:
            return torch.complex(self.U_real[index], self.U_imag[index])
        return self.U[index]
    
    @torch._dynamo.disable
    def get_range(self, targets=None, indices=None, pad_to=None, pad_mode='border'):
        if targets is not None and indices is not None:
            raise ValueError("Cannot specify both targets and indices")
        elif targets is not None:
            if len(targets) != 2:
                raise ValueError("targets must be a (start, end) pair")
            start_t, end_t = targets
            start_idx = int(float(start_t) * float(self.target_dim))
            end_idx = int(float(end_t) * float(self.target_dim))
            indices = (start_idx, end_idx)
        if indices is None:
            raise ValueError("Must specify targets or indices")

        if len(indices) != 2:
            raise ValueError("indices/targets must be a (start, end) pair")

        start_idx, end_idx = indices
        start_idx = max(0, min(int(start_idx), self.target_dim - 1))
        end_idx = max(0, min(int(end_idx), self.target_dim - 1))

        end_exclusive = end_idx + 1
        if self.is_complex:
            chunk = torch.complex(self.U_real[start_idx:end_exclusive], self.U_imag[start_idx:end_exclusive])
        else:
            chunk = self.U[start_idx:end_exclusive]

        if pad_to is not None and pad_to > chunk.shape[0]:
            needed = pad_to - chunk.shape[0]
            # allow right-pad only if we reached the end index
            if end_idx == self.target_dim - 1:
                if pad_mode == 'border':
                    pad_chunk = chunk[-1:].expand(needed, -1)
                elif pad_mode == 'zeros':
                    pad_chunk = torch.zeros((needed, chunk.shape[1]), device=chunk.device, dtype=chunk.dtype)
                else:
                    raise ValueError("Invalid pad_mode")
                chunk = torch.cat([chunk, pad_chunk], dim=0)
            # allow left-pad only if we started at index 0
            elif start_idx == 0:
                if pad_mode == 'border':
                    pad_chunk = chunk[:1].expand(needed, -1)
                elif pad_mode == 'zeros':
                    pad_chunk = torch.zeros((needed, chunk.shape[1]), device=chunk.device, dtype=chunk.dtype)
                else:
                    raise ValueError("Invalid pad_mode")
                chunk = torch.cat([pad_chunk, chunk], dim=0)
            else:
                raise ValueError("Padding invalid: can only left-pad if start index is 0 or right-pad if end index reaches target_dim-1")
        elif pad_to is not None and pad_to < chunk.shape[0]:
            pad_to = int(pad_to)
            if pad_to <= 0:
                raise ValueError("pad_to must be a positive integer")
            cur_len = int(chunk.shape[0])
            center = cur_len // 2
            half = pad_to // 2
            start = max(0, center - half)
            end = start + pad_to
            if end > cur_len:
                end = cur_len
                start = max(0, end - pad_to)
            chunk = chunk[start:end]
        return chunk


class RealTucker(nn.Module):
    def __init__(self, target_shape, ranks, device='cuda'):
        """Initialize the real-domain Tucker reconstruction branch.

        Args:
            target_shape: Full ``[C, H, W, T]`` shape of the target video tensor.
            ranks: Tucker ranks for channel, height, width, and time factors.
            device: Device on which to allocate the factor parameters.
        """
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.rC, self.rH, self.rW, self.rT = ranks

        self.UH = SimpleTuckerFactor(self.H, self.rH, is_complex=False, device=device)
        self.UW = SimpleTuckerFactor(self.W, self.rW, is_complex=False, device=device)
        self.UC = SimpleTuckerFactor(self.C, self.rC, is_complex=False, device=device)
        self.UT = SimpleTuckerFactor(self.T, self.rT, is_complex=False, device=device)

        self.G = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)

    def forward(self, **kwargs):
        UT = self.UT.get_range(**kwargs)
        UC = self.UC()
        UH = self.UH()
        UW = self.UW()
        return tucker_construct(UT, UC, UH, UW, self.G).contiguous()


class ComplexTucker(RealTucker):
    def __init__(self, target_shape, ranks, grid_res, device='cuda'):
        """Initialize the FFT-domain Tucker reconstruction branch.

        Args:
            target_shape: Full ``[C, H, W, T]`` shape of the target video tensor.
            ranks: Tucker ranks for channel, height, width, and time factors.
            grid_res: Feature grid resolution ``[C, H, W, T]`` used for sampling.
            device: Device on which to allocate the factor parameters.
        """
        super().__init__(target_shape, ranks, device=device)
        self.half_W = (self.W // 2) + 1
        self.UH = SimpleTuckerFactor(self.H, self.rH, is_complex=True, device=device)
        self.UW = SimpleTuckerFactor(self.half_W, self.rW, is_complex=True, device=device)
        self.UC = SimpleTuckerFactor(self.C, self.rC, is_complex=True, device=device)
        self.UT = SimpleTuckerFactor(self.T, self.rT, is_complex=True, device=device)

        self.G = None  # override parent
        self.G_real = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)
        self.G_imag = nn.Parameter(torch.zeros(self.rT, self.rC, self.rH, self.rW, device=device))

        grid_c, grid_h, grid_w = grid_res
        half_grid_w = (grid_w // 2) + 1
        print(f"target shape: C={self.C}, H={self.H}, W={self.W}, T={self.T}")
        print(f"Initializing complex tucker with ranks: C={self.rC}, H={self.rH}, W={self.rW}, T={self.rT}")
        print(f"Initializing feature grid with resolution: C={grid_c}, H={grid_h}, W={grid_w}")

        self.feature_grid = FeatureGrid([self.C * 2, self.H, self.half_W, self.T], grid_res=[grid_c, grid_h, half_grid_w], device=device)

    def forward(self, t, zero_complex_tucker=False, zero_complex_grid=False, **kwargs):
        """Reconstruct frames from complex Tucker factors and the complex feature grid.

        Args:
            t: Scalar or tensor of frame coordinates to reconstruct.
            zero_complex_tucker: Whether to disable the complex Tucker branch.
            zero_complex_grid: Whether to disable the complex feature-grid modulation.
            kwargs: passed to UT get_range

        Returns:
            Real-valued frames reconstructed from the complex-domain representation.
        """
        construct = torch.zeros((t.shape[0], self.C, self.H, self.half_W), device=t.device, dtype=torch.complex64)
        B = kwargs.get('pad_to', None)
        if not zero_complex_tucker:
            UH = self.UH()
            UW = self.UW()
            UC = self.UC()
            UT = self.UT.get_range(**kwargs)
            G = torch.complex(self.G_real, self.G_imag)
            construct = tucker_construct(UT, UC, UH, UW, G)

        if not zero_complex_grid:
            grid = self.feature_grid(B)
            complex_grid = torch.complex(*grid.chunk(2, dim=1))
            if zero_complex_tucker:
                 construct = complex_grid
            else:
                if construct.shape[0] == 6:
                    print(f"kwargs: {kwargs}")
                construct = construct * complex_grid
        real_tucker = torch.fft.irfft2(construct, norm='ortho').real
        return real_tucker.contiguous()


def grid_sample_base(H, W, device):
    """Build a normalized spatial sampling grid for ``grid_sample`` operations.

    Args:
        H: Output grid height.
        W: Output grid width.
        device: Device on which to allocate the grid tensor.

    Returns:
        A tensor shaped ``(H, W, 2)`` containing normalized ``(x, y)`` coordinates.
    """
    y_lin = torch.arange(0, H, device=device)
    x_lin = torch.arange(0, W, device=device)
    y_norm = 2.0 * (y_lin / (H - 1)) - 1.0
    x_norm = 2.0 * (x_lin / (W - 1)) - 1.0
    y, x = torch.meshgrid(y_norm, x_norm, indexing='ij')  # [H, W]
    return torch.stack((x, y), dim=-1)  # [H, W, 2]


class FeatureGrid(nn.Module):
    def __init__(self, target_shape, grid_res, zero_init=False, device="cuda"):
        """Initialize the learned 4D feature grid used as a spatial prior.

        Args:
            target_shape: Full ``[C, H, W, T]`` shape of the target video tensor.
            grid_res: Feature grid resolution ``[C, H, W, T]`` used for sampling.
            zero_init: Unused flag retained for compatibility with older experiments.
            device: Device on which to allocate the grid parameters.
        """
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.grid_c = grid_res[0]
        self.grid_h = grid_res[1]
        self.grid_w = grid_res[2]

        self.grid = nn.Parameter(torch.randn(1, self.grid_c, self.grid_h, self.grid_w, device=device) * 1e-2)
        if self.grid_c != self.C:
            self.channel_proj = nn.Linear(self.grid_c, self.C, bias=True).to(device)
            nn.init.normal_(self.channel_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.channel_proj.bias)
        self.register_buffer(
            "_xy_base",
            grid_sample_base(self.H, self.W, device=device).unsqueeze(0),  # [1, H, W, 2]
            persistent=False
        )

        self._grid_5d_view = None

    def forward(self, B):
        device = self.grid.device
        sample_grid2 = self._xy_base.expand(B, -1, -1, -1)  # [B, H_out, W_out, 2]

        if hasattr(self, 'channel_proj'):
            proj = self.channel_proj(self.grid.permute(0, 2, 3, 1))  # -> [1, H_g, W_g, C]
            grid_4d = proj.permute(0, 3, 1, 2).expand(B, -1, -1, -1)  # -> [B, C, H_g, W_g]
        else:
            grid_4d = self.grid.expand(B, -1, -1, -1)  # [B, grid_c, H_g, W_g]

        sampled = F.grid_sample(
            grid_4d,            # [B, C_or_grid_c, H_g, W_g]
            sample_grid2,       # [B, H_out, W_out, 2]
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        )  # -> [B, C_or_grid_c, H_out, W_out]

        return sampled.contiguous()


def tucker_construct(UT, UC, UH, UW, G):
    """Assemble a Tucker reconstruction from factor matrices and a core tensor.

    Args:
        UT: Temporal factor values for the requested batch.
        UC: Channel factor matrix.
        UH: Height factor matrix.
        UW: Width factor matrix.
        G: Tucker core tensor.

    Returns:
        Reconstructed tensor shaped ``(T, C, H, W)`` for the requested times.
    """
    UT = UT.contiguous()
    UC = UC.contiguous()
    UH = UH.contiguous()
    UW = UW.contiguous()
    G = G.contiguous()

    def _col_norm(M, eps=1e-8):
        """Normalize factor columns to stabilize the Tucker contraction.

        Args:
            M: Real or complex factor matrix.
            eps: Numerical floor added to column norms.

        Returns:
            The column-normalized factor matrix.
        """
        if torch.is_complex(M):
            norms_sq = (M.real**2 + M.imag**2).sum(dim=0, keepdim=True)
        else:
            norms_sq = (M * M).sum(dim=0, keepdim=True)
        norms = torch.sqrt(norms_sq + eps)
        return M / norms

    UH = _col_norm(UH)
    UW = _col_norm(UW)
    UC = _col_norm(UC)
    UT = _col_norm(UT)

    X = torch.tensordot(UC, G, dims=([1], [1]))  # [C, rT, rH, rW]
    X = torch.tensordot(UT, X, dims=([1], [1]))  # [T, C, rH, rW]
    X = torch.tensordot(UH, X, dims=([1], [2]))  # [H, T, C, rW]
    X = torch.tensordot(UW, X, dims=([1], [3]))  # [W, H, T, C]
    X = X.permute(2, 3, 1, 0).contiguous()  # [T, C, H, W]
    return X


class BasicUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda'):
        """Initialize the lightweight refinement and upsampling CNN.

        Args:
            in_channels: Number of channels in the low-resolution input tensor.
            out_channels: Number of channels to emit after upsampling.
            hidden: Hidden channel width used inside the CNN.
            k: Pixel-shuffle upscale factor.
            encoding_len: Unused compatibility parameter retained in the signature.
            device: Device on which to allocate the module.
        """
        super().__init__()
        half_k = k // 2
        self.k = k

        self.upres = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)

        #kaiming init
        for m in self.upres.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Upsample and refine a low-resolution feature tensor.

        Args:
            x: Low-resolution feature tensor to upsample.

        Returns:
            The refined high-resolution output tensor.
        """
        base = self.upres(x)
        return base


class ConvOperator(nn.Module):
    def __init__(self, in_channels, out_channels, h_dim, encoding_len=128, device='cuda'):
        """Initialize a temporally modulated convolutional operator.

        Args:
            in_channels: Number of channels in the operator input tensor.
            out_channels: Number of channels to emit.
            h_dim: Hidden width used inside the operator network.
            encoding_len: Width of the temporal Fourier embedding.
            device: Device on which to allocate the module.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.operator_head = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=1),
        ).to(device)

        self.operator_tail = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, out_channels, kernel_size=1),
        ).to(device)

        self.encoding = FourierEncoding(
            target_dim=encoding_len,
            max_freq=64,
            freq_init="log",
            device=device
        )

        self.t_modulator = nn.Sequential(
            nn.Linear(encoding_len, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2 * h_dim),
        ).to(device)

        nn.init.zeros_(self.operator_tail[-1].weight)
        nn.init.zeros_(self.operator_tail[-1].bias)
        nn.init.zeros_(self.t_modulator[-1].weight)
        nn.init.zeros_(self.t_modulator[-1].bias)

    def forward(self, x, t):
        """Apply the operator to neighboring-frame features at time ``t``.

        Args:
            x: Concatenated feature tensor containing source and target context.
            t: Normalized time coordinates used to modulate the operator.

        Returns:
            Operator residuals aligned with the current frame representation.
        """
        initial = self.operator_head(x)
        t_tensor = torch.as_tensor(t, device=x.device, dtype=torch.float32)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.view(1, 1)
        elif t_tensor.dim() == 1:
            t_tensor = t_tensor.view(-1, 1)
        time_emb = self.encoding(t_tensor)
        modulation = self.t_modulator(time_emb)
        gamma, beta = modulation.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.operator_head[-1].out_channels, 1, 1)
        beta = beta.view(-1, self.operator_head[-1].out_channels, 1, 1)
        modulated = initial * (1 + gamma) + beta
        conv_x = self.operator_tail(modulated)
        return conv_x


class NikaBlock(nn.Module):
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, grid_ranks, conv_hidden, out_channels, device):
        """Initialize the full Nika model block used for training and inference.

        Args:
            target_shape: Full ``[C, H, W, T]`` shape of the target video tensor.
            k: Spatial downsampling and pixel-shuffle upsampling factor.
            real_tucker_ranks: Tucker ranks for the real-domain branch.
            complex_tucker_ranks: Tucker ranks for the FFT-domain branch.
            grid_ranks: Feature-grid resolution for the learned grid branch.
            conv_hidden: Hidden width for the final upsampling CNN.
            out_channels: Number of image channels to predict.
            device: Device on which to allocate the module.
        """
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.dT = 1.0 / (self.T - 1)
        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        )
        # self.real_tucker = torch.compile(self.real_tucker)

        self.grid_features = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        )
        # self.grid_features = torch.compile(self.grid_features)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            grid_res=grid_ranks,
            device=device,
        )

        self.n_heads = 3

        self.groupnorm = nn.GroupNorm(num_groups=self.n_heads, num_channels=self.n_heads * self.C).to(device)
        # self.groupnorm = torch.compile(self.groupnorm)

        op_hdim = 64
        self.operator_steps = 2
        self.B = self.operator_steps * 2 + 1

        self.register_buffer(
            "_zero_base",
            torch.zeros((self.B, self.C, self.H, self.W), device=device),
            persistent=False
        )

        self.flow_operator = ConvOperator(
            in_channels = 5 * self.n_heads * self.C,
            out_channels = self.n_heads * self.C,
            h_dim = op_hdim,
            device = device,
        )

        self.upres = BasicUpres(
            in_channels = self.n_heads * self.C,
            out_channels = out_channels,
            hidden = conv_hidden,
            k = k,    
            device = device,
        )
        # self.upres = torch.compile(self.upres)

        self.log_stats()

    def log_stats(self):
        """Print a parameter-count breakdown for the compiled Nika model."""
        real_tucker_params = sum(p.numel() for p in self.real_tucker.parameters())
        complex_tucker_params = sum(p.numel() for p in self.complex_tucker.parameters())
        grid_params = sum(p.numel() for p in self.grid_features.parameters())
        upres_params = sum(p.numel() for p in self.upres.parameters())
        operator_params = sum(p.numel() for p in self.flow_operator.parameters()) # + sum(p.numel() for p in self.forward_operators.parameters()) + sum(p.numel() for p in self.backward_operators.parameters())
        # operator_params = sum(p.numel() for p in self.forward_operators.parameters()) + sum(p.numel() for p in self.backward_operators.parameters())
        total_params = real_tucker_params + complex_tucker_params + grid_params + upres_params + operator_params
        print(f"NikaBlock parameters:")
        print(f"  Real Tucker:     {real_tucker_params / 1e6:.3f}M")
        print(f"  Complex Tucker:  {complex_tucker_params / 1e6:.3f}M")
        print(f"  Feature Grid:    {grid_params / 1e6:.3f}M")
        print(f"  Flow Operator:   {operator_params / 1e6:.3f}M")
        # print(f"  Forward Operator:{sum(p.numel() for p in self.forward_operators.parameters()) / 1e6:.3f}M")
        # print(f"  Backward Operator:{sum(p.numel() for p in self.backward_operators.parameters()) / 1e6:.3f}M")
        print(f"  Upsampling CNN:  {upres_params / 1e6:.3f}M")
        print(f"  Total:           {total_params / 1e6:.3f}M")

    def forward(self, norm_t, noise_op=None, zero_real_tucker=False, zero_complex_tucker=False, zero_feature_grid=False, zero_complex_grid=False, return_operators=False):
        """Predict frames at normalized time coordinates with optional ablations.

        Args:
            norm_t: Scalar or tensor of normalized frame coordinates in ``[0, 1]``.
            noise_op: Unused compatibility argument retained in the signature.
            zero_real_tucker: Whether to disable the real-domain Tucker branch.
            zero_complex_tucker: Whether to disable the FFT-domain Tucker branch.
            zero_feature_grid: Whether to disable the real feature-grid branch.
            zero_complex_grid: Whether to disable complex feature-grid modulation.
            return_operators: Whether to also return upsampled operator residuals.

        Returns:
            Either the reconstructed frames alone or the frames plus operator terms.
        """
        if type(norm_t) is not torch.Tensor:
            norm_t = torch.tensor([norm_t], device=self.grid_features.grid.device, dtype=torch.float32)

        min_t = torch.max(torch.tensor(0.0, device=norm_t.device), norm_t.min() - self.dT * self.operator_steps)
        max_t = torch.min(torch.tensor(1.0, device=norm_t.device), norm_t.max() + self.dT * self.operator_steps)

        if not zero_real_tucker:
            real_tucker = self.real_tucker(targets=(min_t, max_t), pad_to=self.B)
        else:
            real_tucker = self._zero_base.expand(self.B, -1, -1, -1)

        if not zero_feature_grid:
            grid_features = self.grid_features(self.B)
        else:
            grid_features = self._zero_base.expand(self.B, -1, -1, -1)

        if not zero_complex_tucker or not zero_complex_grid:
            complex_tucker = self.complex_tucker(
                zero_complex_tucker=zero_complex_tucker, zero_complex_grid=zero_complex_grid, targets=(min_t, max_t), pad_to=self.B
            )
        else:
            complex_tucker = self._zero_base.expand(self.B, -1, -1, -1)

        response = self.groupnorm(
            torch.cat([real_tucker, grid_features, complex_tucker], dim=1)
        )

        aggregated = response[self.operator_steps]

        B, Cc, H, W = response.shape
        # combine the time (B) and channel (Cc) dims into a single channel dim -> [1, B*Cc, H, W]
        op_input = response.reshape(1, B * Cc, H, W)
        prediction = self.flow_operator(op_input, norm_t)
        aggregated = aggregated.unsqueeze(0)
        aggregated = aggregated + prediction

        if return_operators:
            raise NotImplementedError("Operator return not implemented in this version")
        #     operator_steps = []

        prediction = self.upres(aggregated)

        # if return_operators:
        #     operator_residuals = []
        #     for op in operator_steps:
        #         operator_residuals.append(self.upres(op))
        #     return prediction, *operator_residuals
        return prediction

    def test_images(self, output_dir):
        """Render a few sample frames and write them to disk for spot checks.

        Args:
            output_dir: Directory where preview frames should be written.
        """
        # self.eval()
        with torch.no_grad():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            rand_vals = torch.linspace(0, 1, steps=10, dtype=torch.float32, device=self.grid_features.grid.device)

            # Warmup once to get output shape (not measured)
            target0 = torch.tensor([rand_vals[0]], device=self.grid_features.grid.device, dtype=torch.float32)
            img0 = self.forward(target0)
            # allocate GPU buffer to avoid appends / CPU transfers during timing
            imgs_gpu = torch.empty((len(rand_vals),) + img0.shape, device=img0.device, dtype=img0.dtype)
            imgs_gpu[0].copy_(img0.detach())

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            # timed region: only forward calls and storing into preallocated GPU buffer
            starter.record()
            for i, rv in enumerate(rand_vals):
                if i == 0:
                    continue  # already have warmup result stored
                torch.compiler.cudagraph_mark_step_begin()
                target = torch.tensor([rv], device=self.grid_features.grid.device, dtype=torch.float32)
                img = self.forward(target)
                imgs_gpu[i].copy_(img.detach())
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
            average_frame_time = (total_ms / rand_vals.shape[0]) / 1000.0
            print(f"Average inference time per frame: {average_frame_time:.5f}s")
            print(f"FPS: {1.0 / average_frame_time:.2f}")

            # Move to CPU and save (post-measurement)
            imgs_cpu = imgs_gpu.clamp(0.0, 1.0).cpu()
            for i in range(len(rand_vals)):
                save_image(imgs_cpu[i], f"{output_dir}/frame_{i:04d}.png")


def feature_test(vid, name, config, device):
    """Train the main Nika model on one video sequence.

    Args:
        vid: Video tensor shaped ``(T, C, H, W)`` containing training frames.
        name: Name of the sequence, used for checkpoint naming.
        config: Key into ``REFERENCES`` selecting the model hyperparameters.
        device: Device on which to run training.
    """
    batch_size = 1

    model_kwargs = REFERENCES[config]
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        device=device,
    )

    model = torch.compile(model)

    base_lr = 1e-2
    # single optimizer for all parameters (everything moves together)
    opt = SOAP(model.parameters(), lr=base_lr, weight_decay=0)

    # Tapered warm restarts: CosineAnnealingWarmRestarts combined with
    # a global linear taper multiplier so restart amplitudes shrink
    # over the course of training (prevents large unrecoverable spikes).
    class TaperedWarmRestarts:
        def __init__(self, optimizer, T_0=200, T_mult=2, eta_min=0.0, max_epochs=2000, final_multiplier=0.2):
            """Initialize a warm-restart scheduler whose amplitude tapers over time.

            Args:
                optimizer: Optimizer whose learning rate should be scheduled.
                T_0: Initial restart period in epochs.
                T_mult: Multiplicative factor applied to later restart periods.
                eta_min: Minimum cosine-annealed learning rate.
                max_epochs: Epoch budget used to compute the taper multiplier.
                final_multiplier: Final scaling factor applied at the end of training.
            """
            self.optimizer = optimizer
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            self.max_epochs = max_epochs
            self.final_multiplier = float(final_multiplier)
            # store original base lrs to compute taper relative to initial scale
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            self.last_epoch = -1

        def step(self, epoch=None):
            """Advance the scheduler and apply the taper-adjusted restart scale.

            Args:
                epoch: Optional explicit epoch index to step to.
            """
            # advance epoch counter
            if epoch is None:
                self.last_epoch += 1
                epoch = self.last_epoch
            else:
                self.last_epoch = int(epoch)

            # step the underlying cosine-with-restarts scheduler
            self.base_scheduler.step(epoch)

            # compute a linear taper from 1.0 -> final_multiplier over max_epochs
            taper = 1.0 - (float(self.last_epoch) / float(self.max_epochs)) * (1.0 - self.final_multiplier)
            if taper < self.final_multiplier:
                taper = self.final_multiplier

            # the base_scheduler already updated optimizer.param_groups' lr values
            # relative to the stored base_lrs; compute cosine scale and reapply with taper
            for i, group in enumerate(self.optimizer.param_groups):
                base = float(self.base_lrs[i])
                # avoid division by zero
                cos_lr = float(group['lr'])
                cos_scale = (cos_lr / base) if base > 0.0 else 1.0
                group['lr'] = base * cos_scale * taper

    # instantiate tapered scheduler
    scheduler = TaperedWarmRestarts(
        opt,
        T_0=200,
        T_mult=2,
        eta_min=base_lr * 0.1,
        max_epochs=3000,
        final_multiplier=0.2,
    )

    best_psnr = float('-inf')
    best_epoch = -1

    for epoch in range(3000):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        start_time = time.time()
        num_batches = (vid.shape[0] + batch_size - 1) // batch_size
        for t in range(num_batches):
            min_t = t * batch_size
            max_t = min((t + 1) * batch_size, vid.shape[0])
            batch_gt = vid[min_t:max_t].to(torch.float32) / 255.0
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
            norm_t_batch = t_batch.float() / (vid.shape[0] - 1)
            torch.compiler.cudagraph_mark_step_begin()
            prediction = model(norm_t_batch)
            mse = F.mse_loss(prediction, batch_gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            frame_loss = (-psnr).mean() / num_batches
            frame_loss.backward()
            loss += frame_loss.item()
        opt.step()
        scheduler.step()
        average_frame_time = (time.time() - start_time) / vid.shape[0]
        epoch_psnr = -loss
        print(f"Epoch {epoch} loss: {loss:.4f}, time: {average_frame_time:.5f}s, PSNR: {epoch_psnr:.2f}")

        if epoch_psnr > best_psnr and (epoch - best_epoch >= 10 or best_epoch == -1):
            best_psnr = epoch_psnr
            best_epoch = epoch
            model_path = f"models/{config}-{name}-epoch{epoch}-psnr{best_psnr:.2f}.torch"
            torch.save(model.state_dict(), model_path)
            os.sync()
            print(f"New best model saved at epoch {epoch} with PSNR: {best_psnr:.2f}")

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Tucker PSNR: {epoch_psnr:.2f}")
            model.test_images("out_feature_test")

    print(f"Best PSNR achieved: {best_psnr:.2f} at epoch {best_epoch}")


if __name__ == "__main__":
    device = "cuda:0"
    name = "bunny"
    torch.manual_seed(42)
    vid = load_video_frames(f"static/benchmarks/{name}", device, max_frames=132, dtype=torch.uint8, normalize=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    feature_test(vid, name, f"small", device=device)
