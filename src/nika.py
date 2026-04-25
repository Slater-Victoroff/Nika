"""Core Nika model definitions, training entrypoints, and visualization helpers."""

import os
import time
import math
import glob
from functools import partial

import torch
from torch.profiler import profile, ProfilerActivity, record_function
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F

import numpy as np
import imageio.v3 as iio
from PIL import Image
import argparse
import concurrent.futures

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
            start_idx = int(float(torch.as_tensor(start_t).detach()) * float(self.target_dim))
            end_idx = int(float(torch.as_tensor(end_t).detach()) * float(self.target_dim))
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

    def forward(self, zero_complex_tucker=False, zero_complex_grid=False, **kwargs):
        """Reconstruct frames from complex Tucker factors and the complex feature grid.

        Args:
            t: Scalar or tensor of frame coordinates to reconstruct.
            zero_complex_tucker: Whether to disable the complex Tucker branch.
            zero_complex_grid: Whether to disable the complex feature-grid modulation.
            kwargs: passed to UT get_range

        Returns:
            Real-valued frames reconstructed from the complex-domain representation.
        """
        if zero_complex_tucker and zero_complex_grid:
            construct = torch.zeros((self.C, self.H, self.half_W), device=t.device, dtype=torch.complex64)
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
        self.channel_proj = nn.Conv2d(self.grid_c, self.C, kernel_size=1, bias=True).to(device)
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

        with record_function("nika/feature_grid/channel_proj"):
            proj = self.channel_proj(self.grid)  # -> [1, C, H_g, W_g]
            grid_4d = proj.expand(B, -1, -1, -1)  # -> [B, C, H_g, W_g]

        with record_function("nika/feature_grid/grid_sample"):
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

    with record_function("nika/tucker_construct"):
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

class ConvNextUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda', expansion=2):
        """Initialize a lightweight ConvNeXt-style upsampling block.

        Args:
            in_channels: Number of channels in the low-resolution input tensor.
            out_channels: Number of channels to emit after upsampling.
            hidden: Hidden channel width used inside the CNN.
            k: Pixel-shuffle upscale factor.
            encoding_len: Unused compatibility parameter retained in the signature.
            device: Device on which to allocate the module.
            expansion: Expansion ratio inside the ConvNeXt-lite residual block.
        """
        super().__init__()
        self.k = k
        inner_dim = hidden * expansion

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
        ).to(device)

        # ConvNeXt-lite residual block
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
        ).to(device)

        # Cheap channels-first stand-in for ConvNeXt LayerNorm
        self.norm = nn.GroupNorm(hidden // 4, hidden).to(device)
        # self.norm = nn.Identity()

        self.pw_expand = nn.Conv2d(hidden, inner_dim, kernel_size=1).to(device)
        self.act = nn.GELU()
        self.pw_contract = nn.Conv2d(inner_dim, hidden, kernel_size=1).to(device)

        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)

        # Kaiming init
        for m in self.modules():
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
        with record_function("nika/upres/in_proj"):
            x = self.in_proj(x)

        with record_function("nika/upres/residual_block"):
            residual = x
            x = self.dwconv(x)
            x = self.norm(x)
            x = self.pw_expand(x)
            x = self.act(x)
            x = self.pw_contract(x)
            x = x + residual

        with record_function("nika/upres/out_proj"):
            x = self.out_proj(x)
        return x


class ConvOperator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_slots,
        h_dim,
        slot_rank=2,
        encoding_len=128,
        device="cuda",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_slots = n_slots
        self.slot_rank = slot_rank

        # Per-channel slot mixing: each logical channel mixes its 5 temporal slots
        # into slot_rank learned combinations.
        self.slot_proj = nn.Conv2d(
            in_channels=n_slots * in_channels,
            out_channels=slot_rank * in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=True,
        ).to(device)

        mixed_channels = slot_rank * in_channels
        self.operator_head = nn.Sequential(
            nn.Conv2d(mixed_channels, h_dim, kernel_size=1),
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
            max_freq=128,
            freq_init="log",
            init_mode="random",
            device=device
        )

        self.t_modulator1 = nn.Sequential(
            nn.Linear(encoding_len, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2 * h_dim),
        ).to(device)

        self.t_modulator2 = nn.Sequential(
            nn.Linear(encoding_len, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2 * h_dim),
        ).to(device)

        nn.init.zeros_(self.operator_tail[-1].weight)
        nn.init.zeros_(self.operator_tail[-1].bias)
        nn.init.zeros_(self.t_modulator1[-1].weight)
        nn.init.zeros_(self.t_modulator1[-1].bias)
        nn.init.zeros_(self.t_modulator2[-1].weight)
        nn.init.zeros_(self.t_modulator2[-1].bias)

    def _time_to_embedding(self, t, device):
        t_tensor = torch.as_tensor(t, device=device, dtype=torch.float32)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.view(1, 1)
        elif t_tensor.dim() == 1:
            t_tensor = t_tensor.view(-1, 1)
        return self.encoding(t_tensor)

    def _apply_film(self, x, modulation):
        gamma, beta = modulation.chunk(2, dim=-1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)
        return x * (1 + gamma) + beta

    def forward(self, x, t):
        """
        x: [N_slots, C, H, W]
        returns: [1, out_channels, H, W]
        """
        n_slots, c, h, w = x.shape
        assert n_slots == self.n_slots
        assert c == self.in_channels

        # Flatten slot axis into channels, but preserve structure via grouped slot mixing
        x = x.reshape(1, n_slots * c, h, w)

        # Cheap learned temporal basis extraction
        with record_function("nika/flow_operator/slot_proj"):
            x = self.slot_proj(x)

        with record_function("nika/flow_operator/head"):
            h1 = self.operator_head(x)

        with record_function("nika/flow_operator/modulation1"):
            time_emb = self._time_to_embedding(t, h1.device)
            h1 = self._apply_film(h1, self.t_modulator1(time_emb))

        with record_function("nika/flow_operator/tail"):
            h2 = self.operator_tail[:-1](h1)
            h2 = self._apply_film(h2, self.t_modulator2(time_emb))
            out = self.operator_tail[-1](h2)
        return out


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
            in_channels = self.n_heads * self.C,
            out_channels = self.n_heads * self.C,
            n_slots = self.B,
            h_dim = op_hdim,
            device = device,
        )

        self.upres = ConvNextUpres(
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

        with record_function("nika/real_tucker"):
            if not zero_real_tucker:
                real_tucker = self.real_tucker(targets=(min_t, max_t), pad_to=self.B)
            else:
                real_tucker = self._zero_base.expand(self.B, -1, -1, -1)

        with record_function("nika/real_grid"):
            if not zero_feature_grid:
                grid_features = self.grid_features(self.B)
            else:
                grid_features = self._zero_base.expand(self.B, -1, -1, -1)

        with record_function("nika/complex_tucker"):
            if not zero_complex_tucker or not zero_complex_grid:
                complex_tucker = self.complex_tucker(
                    zero_complex_tucker=zero_complex_tucker, zero_complex_grid=zero_complex_grid, targets=(min_t, max_t), pad_to=self.B
                )
            else:
                complex_tucker = self._zero_base.expand(self.B, -1, -1, -1)

        with record_function("nika/feature_fusion"):
            response = self.groupnorm(
                torch.cat([real_tucker, grid_features, complex_tucker], dim=1)
            )

        aggregated = response[self.operator_steps]

        with record_function("nika/flow_operator"):
            prediction = self.flow_operator(response, norm_t)
        aggregated = aggregated.unsqueeze(0)
        aggregated = aggregated + prediction

        if return_operators:
            raise NotImplementedError("Operator return not implemented in this version")
        #     operator_steps = []

        with record_function("nika/upres"):
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
            batch_size = 2

            # Warmup once to get output shape (not measured)
            target0 = torch.tensor([rand_vals[0]], device=self.grid_features.grid.device, dtype=torch.float32)
            img0 = self.forward(target0)
            # allocate GPU buffer to avoid appends / CPU transfers during timing
            imgs_gpu = torch.empty((len(rand_vals),) + img0.shape[1:], device=img0.device, dtype=img0.dtype)
            imgs_gpu[0].copy_(img0[0].detach())

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            # timed region: batch forward calls and storing into preallocated GPU buffer
            starter.record()
            # iterate in batch-sized chunks, skip first value (warmup) which is already stored
            for i in range(0, len(rand_vals), batch_size):
                if i == 0:
                    # already computed and stored warmup index 0
                    continue
                batch_vals = rand_vals[i:i+batch_size]
                torch.compiler.cudagraph_mark_step_begin()
                out = self.forward(batch_vals)
                # out shape: [B, C, H, W]
                for j in range(out.shape[0]):
                    imgs_gpu[i + j].copy_(out[j].detach())
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
    # increase batch_size to reduce number of small kernel launches
    # (processing multiple frames per forward/backward can improve GPU utilization)
    batch_size = 2

    model_kwargs = REFERENCES[config]
    model = NikaBlock(
        target_shape=[4, vid.shape[2], vid.shape[3], vid.shape[0]],
        k=4,
        **model_kwargs,
        out_channels=3,
        device=device,
    )

    model = torch.compile(model)

    base_lr = 5e-3
    # single optimizer for all parameters (everything moves together)
    opt = SOAP(model.parameters(), lr=base_lr, weight_decay=0)

    # Tapered warm restarts: custom cosine annealing with restarts combined with
    # a global linear taper multiplier so restart amplitudes shrink
    # over the course of training (prevents large unrecoverable spikes).
    class TaperedWarmRestarts:
        def __init__(self, optimizer, T_0=200, T_mult=2.0, eta_min=0.0, max_epochs=2000, final_multiplier=0.2, hold_epochs=0):
            """Initialize a warm-restart scheduler whose amplitude tapers over time.

            Args:
                optimizer: Optimizer whose learning rate should be scheduled.
                T_0: Initial restart period in epochs.
                T_mult: Multiplicative factor applied to later restart periods (supports float values like 1.5).
                eta_min: Minimum cosine-annealed learning rate.
                max_epochs: Epoch budget used to compute the taper multiplier.
                final_multiplier: Final scaling factor applied at the end of training.
                hold_epochs: Number of initial epochs to keep the base learning rate unchanged.
            """
            self.optimizer = optimizer
            self.T_0 = T_0
            self.T_mult = float(T_mult)
            self.eta_min = eta_min
            self.max_epochs = max_epochs
            self.final_multiplier = float(final_multiplier)
            self.hold_epochs = int(hold_epochs)
            # store original base lrs to compute taper relative to initial scale
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            self.last_epoch = -1

        def _get_cycle_lr(self, epoch):
            """Compute the cosine-annealed learning rate for a given epoch within a warm-restart cycle."""
            # Determine which restart cycle we're in and position within that cycle
            T_cur = 0
            cycle = 0
            while epoch >= T_cur + self.T_0 * (self.T_mult ** cycle):
                T_cur += self.T_0 * (self.T_mult ** cycle)
                cycle += 1
            
            # Compute position within current cycle
            epoch_in_cycle = epoch - T_cur
            cycle_length = self.T_0 * (self.T_mult ** cycle)
            
            # Cosine annealing formula: eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T_i)) / 2
            progress = epoch_in_cycle / cycle_length
            return self.eta_min + (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * progress)) / 2.0

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

            if self.last_epoch < self.hold_epochs:
                for i, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = float(self.base_lrs[i])
                return

            local_epoch = self.last_epoch - self.hold_epochs

            # compute cosine-annealed scale (normalized to [0, 1])
            cos_scale = self._get_cycle_lr(local_epoch)

            # compute a linear taper from 1.0 -> final_multiplier over the post-hold schedule
            taper_progress = float(local_epoch) / float(max(1, self.max_epochs - self.hold_epochs))
            taper = 1.0 - taper_progress * (1.0 - self.final_multiplier)
            if taper < self.final_multiplier:
                taper = self.final_multiplier

            # apply cosine scale and taper to each parameter group
            for i, group in enumerate(self.optimizer.param_groups):
                base = float(self.base_lrs[i])
                group['lr'] = base * cos_scale * taper

    # instantiate tapered scheduler
    # T_mult=2.0 grows cycles as 100->200->400->800 epochs, giving later restarts more room than T_mult=1.
    scheduler = TaperedWarmRestarts(
        opt,
        T_0=100,
        T_mult=2.0,
        eta_min=base_lr * 0.1,
        max_epochs=3000,
        final_multiplier=0.2,
        hold_epochs=200,
    )

    best_psnr = float('-inf')
    best_epoch = -1

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

    use_cuda_timing = isinstance(device, str) and ("cuda" in device.lower())

    for epoch in range(3000):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
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
            prediction = model(norm_t_batch)
            mse = F.mse_loss(prediction, batch_gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            frame_loss = (-psnr).mean() / num_batches
            frame_loss.backward()
            loss += frame_loss.item()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        # epoch timing: total time, per-frame average, and equivalent FPS
        if use_cuda_timing:
            epoch_ender.record()
            torch.cuda.synchronize(device)
            epoch_time = epoch_starter.elapsed_time(epoch_ender) / 1000.0
        else:
            epoch_time = time.time() - start_time
        average_frame_time = epoch_time / float(vid.shape[0])
        fps = (float(vid.shape[0]) / epoch_time) if epoch_time > 0.0 else float('inf')
        epoch_psnr = -loss
        print(
            f"Epoch {epoch} loss: {loss:.4f}, time: {average_frame_time:.5f}s/frame (FPS: {fps:.2f}), PSNR: {epoch_psnr:.2f}"
        )

        if epoch_psnr > best_psnr and (epoch - best_epoch >= 10 or best_epoch == -1):
            best_psnr = epoch_psnr
            best_epoch = epoch
            model_path = f"models/{config}-{name}-epoch{epoch}-psnr{best_psnr:.2f}.torch"
            # move state to CPU on main thread to avoid GPU-side sync inside background thread
            try:
                state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            except Exception:
                state_cpu = None
            # schedule background save (does not block main thread)
            if state_cpu is not None:
                executor.submit(torch.save, state_cpu, model_path)
                # call os.sync in background as well
                executor.submit(lambda p: (os.sync(), print(f"Background saved {p}")), model_path)
            else:
                executor.submit(_save_checkpoint_in_bg, model, model_path, epoch, float(best_psnr))

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Tucker PSNR: {epoch_psnr:.2f}")
            model.test_images("out_feature_test")

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
