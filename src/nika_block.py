"""Core Nika block and operator/upres modules."""

import os
import math

import torch
import torch.nn as nn
from torch.profiler import record_function
from torchvision.utils import save_image

from encoding_utils import FourierEncoding
from logging_utils import log_nika_block_stats
from tucker_modules import RealTucker, ComplexTucker, FeatureGrid


class ConvNextUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda', expansion=2):
        super().__init__()
        self.k = k
        inner_dim = hidden * expansion

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
        ).to(device)

        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
        ).to(device)

        self.norm = nn.GroupNorm(hidden // 4, hidden).to(device)

        self.pw_expand = nn.Conv2d(hidden, inner_dim, kernel_size=1).to(device)
        self.act = nn.GELU()
        self.pw_contract = nn.Conv2d(inner_dim, hidden, kernel_size=1).to(device)

        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
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
            device=device,
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
        if x.dim() == 4:
            n_slots, c, h, w = x.shape
            assert n_slots == self.n_slots
            assert c == self.in_channels
            x = x.reshape(1, n_slots * c, h, w)
        elif x.dim() == 5:
            b, n_slots, c, h, w = x.shape
            assert n_slots == self.n_slots
            assert c == self.in_channels
            x = x.reshape(b, n_slots * c, h, w)
        else:
            raise ValueError("x must have shape [n_slots, c, h, w] or [b, n_slots, c, h, w]")

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
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, base_grid_channels, conv_hidden, out_channels, operator_steps, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.operator_steps = operator_steps
        self.H = int(self.H // k)
        self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]

        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            op_steps=self.operator_steps,
            device=device,
        )

        self.grid_features = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=[base_grid_channels, self.H, self.W],
            device=device,
        )

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            base_grid_channels=base_grid_channels,
            op_steps=self.operator_steps,
            device=device,
        )

        self.n_heads = 3

        self.groupnorm = nn.GroupNorm(num_groups=self.n_heads, num_channels=self.n_heads * self.C).to(device)

        op_hdim = 64
        self.B = self.operator_steps * 2 + 1

        self.register_buffer(
            "_zero_base",
            torch.zeros((self.B, self.C, self.H, self.W), device=device),
            persistent=False,
        )

        self.flow_operator = ConvOperator(
            in_channels=self.n_heads * self.C,
            out_channels = self.C,
            # out_channels=self.n_heads * self.C,
            n_slots=self.B,
            h_dim=op_hdim,
            device=device,
        )

        self.upres = ConvNextUpres(
            in_channels=(self.n_heads + 1) * self.C,
            out_channels=out_channels,
            hidden=conv_hidden,
            k=k,
            device=device,
        )

        self.log_stats()

    def log_stats(self):
        log_nika_block_stats(self)

    def _prepare_shared_batch_tensors(
        self,
        zero_real_tucker=False,
        zero_complex_tucker=False,
        zero_feature_grid=False,
        zero_complex_grid=False,
    ):
        shared = {}

        if not zero_real_tucker:
            shared["real_tucker"] = self.real_tucker.prepare_shared()

        if not zero_feature_grid:
            with record_function("nika/real_grid/shared_setup"):
                shared["grid_features"] = self.grid_features(self.B)

        if not zero_complex_tucker or not zero_complex_grid:
            shared["complex_tucker"] = self.complex_tucker.prepare_shared(
                self.B,
                include_tucker=not zero_complex_tucker,
                include_grid=not zero_complex_grid,
            )

        return shared

    def forward(self, norm_t, noise_op=None, zero_real_tucker=False, zero_complex_tucker=False, zero_feature_grid=False, zero_complex_grid=False, return_operators=False):
        if type(norm_t) is not torch.Tensor:
            norm_t = torch.tensor([norm_t], device=self.grid_features.grid.device, dtype=torch.float32)
        else:
            norm_t = norm_t.to(device=self.grid_features.grid.device, dtype=torch.float32)

        if norm_t.dim() == 0:
            norm_t = norm_t.view(1)
        elif norm_t.dim() > 1:
            norm_t = norm_t.reshape(-1)

        shared = self._prepare_shared_batch_tensors(
            zero_real_tucker=zero_real_tucker,
            zero_complex_tucker=zero_complex_tucker,
            zero_feature_grid=zero_feature_grid,
            zero_complex_grid=zero_complex_grid,
        )

        batch_size = int(norm_t.shape[0])

        with record_function("nika/real_tucker"):
            if not zero_real_tucker:
                real_tucker = self.real_tucker(norm_t, shared=shared.get("real_tucker"))
            else:
                real_tucker = self._zero_base.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        with record_function("nika/real_grid"):
            if not zero_feature_grid:
                grid_features = shared.get("grid_features")
                if grid_features is None:
                    grid_features = self.grid_features(self.B)
                grid_features = grid_features.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            else:
                grid_features = self._zero_base.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        with record_function("nika/complex_tucker"):
            if not zero_complex_tucker or not zero_complex_grid:
                complex_tucker = self.complex_tucker(
                    norm_t,
                    zero_complex_tucker=zero_complex_tucker,
                    zero_complex_grid=zero_complex_grid,
                    shared=shared.get("complex_tucker"),
                )
            else:
                complex_tucker = self._zero_base.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        with record_function("nika/feature_fusion"):
            fused = torch.cat([real_tucker, grid_features, complex_tucker], dim=2)
            fused_shape = fused.shape
            fused = fused.reshape(fused_shape[0] * fused_shape[1], fused_shape[2], fused_shape[3], fused_shape[4])
            fused = self.groupnorm(fused)
            response = fused.view(fused_shape[0], fused_shape[1], fused_shape[2], fused_shape[3], fused_shape[4])

        aggregated = response[:, self.operator_steps]

        with record_function("nika/flow_operator"):
            residual = self.flow_operator(response, norm_t)
            aggregated = torch.cat([aggregated, residual], dim=1)
        # aggregated = aggregated + residual

        with record_function("nika/upres"):
            prediction = self.upres(aggregated)

        if return_operators:
            return prediction, residual

        return prediction
