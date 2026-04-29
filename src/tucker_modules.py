"""Tucker factorization and feature-grid modules used by Nika."""

import torch
import torch.nn as nn
from torch.profiler import record_function


class SimpleTuckerFactor(nn.Module):
    def __init__(self, target_dim, rank, is_complex=False, base_mag=1e-2, device='cuda'):
        super().__init__()
        self.is_complex = is_complex
        self.target_dim = target_dim

        if self.is_complex:
            self.U_real = nn.Parameter(torch.randn(self.target_dim, rank, device=device) * base_mag)
            self.U_imag = nn.Parameter(torch.zeros(self.target_dim, rank, device=device))
        else:
            self.U = nn.Parameter(torch.randn(self.target_dim, rank, device=device) * base_mag)
        self.device = device

    def forward(self, indices=None):
        if indices is None:
            if self.is_complex:
                return torch.complex(self.U_real, self.U_imag)
            return self.U
        if self.is_complex:
            return torch.complex(self.U_real[indices], self.U_imag[indices])
        return self.U[indices]


class FeatureGrid(nn.Module):
    def __init__(self, target_shape, grid_res, zero_init=False, device="cuda"):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.grid_c = grid_res[0]
        self.grid_h = grid_res[1]
        self.grid_w = grid_res[2]

        self.grid = nn.Parameter(torch.randn(1, self.grid_c, self.grid_h, self.grid_w, device=device) * 1e-2)
        self.channel_proj = nn.Conv2d(self.grid_c, self.C, kernel_size=1, bias=True).to(device)
        nn.init.normal_(self.channel_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.channel_proj.bias)

    def forward(self, B):
        with record_function("nika/feature_grid/channel_proj"):
            proj = self.channel_proj(self.grid)
        return proj.expand(B, -1, -1, -1).contiguous()


def tucker_construct(UT, UC, UH, UW, G):
    UT = UT.contiguous()
    UC = UC.contiguous()
    UH = UH.contiguous()
    UW = UW.contiguous()
    G = G.contiguous()

    with record_function("nika/tucker_construct"):
        with record_function("nika/tucker_construct/einsum/uc_g"):
            X = torch.einsum('cq,tqhw->cthw', UC, G)
        with record_function("nika/tucker_construct/einsum/ut_x"):
            X = torch.einsum('...kt,cthw->...kchw', UT, X)
        with record_function("nika/tucker_construct/einsum/uh_x"):
            X = torch.einsum('ph,...kchw->...kcpw', UH, X)
        with record_function("nika/tucker_construct/einsum/uw_x"):
            X = torch.einsum('qw,...kcpw->...kcpq', UW, X)
    return X


class RealTucker(nn.Module):
    def __init__(self, target_shape, ranks, op_steps=2, device='cuda:0'):
        super().__init__()
        self.C, self.H, self.W, self.base_T = target_shape
        self.op_steps = int(op_steps)
        self.T = self.base_T + (2 * self.op_steps)
        self.rC, self.rH, self.rW, self.rT = ranks
        self.device = device

        self.UH = SimpleTuckerFactor(self.H, self.rH, is_complex=False, device=device)
        self.UW = SimpleTuckerFactor(self.W, self.rW, is_complex=False, device=device)
        self.UC = SimpleTuckerFactor(self.C, self.rC, is_complex=False, device=device)
        self.UT = SimpleTuckerFactor(self.T, self.rT, is_complex=False, device=device)

        self.G = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)

    def _targets_to_window_indices(self, targets):
        targets_t = torch.as_tensor(targets, device=self.device, dtype=torch.float32)
        if targets_t.dim() == 0:
            targets_t = targets_t.view(1, 1)
        elif targets_t.dim() == 1:
            targets_t = targets_t.view(-1, 1)
        elif not (targets_t.dim() == 2 and targets_t.shape[1] == 1):
            raise ValueError("targets must be scalar, [B], or [B, 1]")

        clamped = targets_t.clamp(0.0, 1.0)
        centers = torch.floor(clamped * float(max(1, self.base_T - 1))).to(torch.long)
        centers = centers + self.op_steps

        offsets = torch.arange(
            -self.op_steps,
            self.op_steps + 1,
            device=targets_t.device,
            dtype=torch.long,
        ).view(1, -1)

        indices = centers + offsets
        return indices.clamp_(0, self.T - 1)

    def prepare_shared(self):
        with record_function("nika/real_tucker/shared_setup"):
            with record_function("nika/real_tucker/shared_setup/factor_lookup"):
                UC = self.UC()
                UH = self.UH()
                UW = self.UW()
        return {"UC": UC, "UH": UH, "UW": UW}

    def forward(self, targets, shared=None):
        with record_function("nika/real_tucker/window_indices"):
            indices = self._targets_to_window_indices(targets)
        if shared is None:
            with record_function("nika/real_tucker/factor_lookup"):
                UT = self.UT(indices)
                UC = self.UC()
                UH = self.UH()
                UW = self.UW()
            result = tucker_construct(UT, UC, UH, UW, self.G)
        else:
            with record_function("nika/real_tucker/factor_lookup"):
                UT = self.UT(indices)
            result = tucker_construct(UT, shared["UC"], shared["UH"], shared["UW"], self.G)
        with record_function("nika/real_tucker/contiguous"):
            return result.contiguous()


class ComplexTucker(RealTucker):
    def __init__(self, target_shape, ranks, base_grid_channels=None, grid_res=None, op_steps=2, device='cuda'):
        super().__init__(target_shape, ranks, op_steps=op_steps, device=device)
        self.half_W = (self.W // 2) + 1
        self.UH = SimpleTuckerFactor(self.H, self.rH, is_complex=True, device=device)
        self.UW = SimpleTuckerFactor(self.half_W, self.rW, is_complex=True, device=device)
        self.UC = SimpleTuckerFactor(self.C, self.rC, is_complex=True, device=device)
        self.UT = SimpleTuckerFactor(self.T, self.rT, is_complex=True, device=device)

        self.G = None
        self.G_real = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)
        self.G_imag = nn.Parameter(torch.zeros(self.rT, self.rC, self.rH, self.rW, device=device))

        if base_grid_channels is None:
            if grid_res is None:
                raise ValueError("ComplexTucker requires base_grid_channels (or legacy grid_res)")
            base_grid_channels = int(grid_res[0])
        base_grid_channels = int(base_grid_channels)

        self.feature_grid = FeatureGrid(
            [self.C * 2, self.H, self.half_W, self.T],
            grid_res=[base_grid_channels, self.H, self.half_W],
            device=device,
        )

    def prepare_shared(self, K, include_tucker=True, include_grid=True):
        shared = {}

        if include_tucker:
            with record_function("nika/complex_tucker/shared_setup"):
                with record_function("nika/complex_tucker/shared_setup/factor_lookup"):
                    UH = self.UH()
                    UW = self.UW()
                    UC = self.UC()
                with record_function("nika/complex_tucker/build_complex_G"):
                    G = torch.complex(self.G_real, self.G_imag)
            shared.update({"UH": UH, "UW": UW, "UC": UC, "G": G})

        if include_grid:
            with record_function("nika/complex_tucker/shared_setup/feature_grid"):
                grid = self.feature_grid(K)
                complex_grid = torch.complex(*grid.chunk(2, dim=1))
            shared["complex_grid"] = complex_grid

        return shared

    def forward(self, targets, zero_complex_tucker=False, zero_complex_grid=False, shared=None):
        with record_function("nika/complex_tucker/window_indices"):
            indices = self._targets_to_window_indices(targets)
        B = int(indices.shape[0])
        K = int(indices.shape[1])

        if zero_complex_tucker and zero_complex_grid:
            construct = torch.zeros((B, K, self.C, self.H, self.half_W), device=self.G_real.device, dtype=torch.complex64)

        if not zero_complex_tucker:
            if shared is None:
                with record_function("nika/complex_tucker/factor_lookup"):
                    UH = self.UH()
                    UW = self.UW()
                    UC = self.UC()
                    UT = self.UT(indices)
                with record_function("nika/complex_tucker/build_complex_G"):
                    G = torch.complex(self.G_real, self.G_imag)
                construct = tucker_construct(UT, UC, UH, UW, G)
            else:
                with record_function("nika/complex_tucker/factor_lookup"):
                    UT = self.UT(indices)
                construct = tucker_construct(
                    UT,
                    shared["UC"],
                    shared["UH"],
                    shared["UW"],
                    shared["G"],
                )

        if not zero_complex_grid:
            if shared is None:
                with record_function("nika/complex_tucker/feature_grid"):
                    grid = self.feature_grid(K)
                    complex_grid = torch.complex(*grid.chunk(2, dim=1))
            else:
                with record_function("nika/complex_tucker/feature_grid"):
                    complex_grid = shared["complex_grid"]
            complex_grid = complex_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
            with record_function("nika/complex_tucker/grid_fuse"):
                if zero_complex_tucker:
                    construct = complex_grid
                else:
                    construct = construct * complex_grid

        with record_function("nika/complex_tucker/irfft2"):
            real_tucker = torch.fft.irfft2(construct, norm='ortho').real
        with record_function("nika/complex_tucker/contiguous"):
            return real_tucker.contiguous()
