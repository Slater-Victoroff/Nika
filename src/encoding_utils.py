"""Positional encoding helpers used by temporal conditioning modules."""

import math

import torch
from torch import nn

class FourierPositionalEncoding(nn.Module):
    """Project coordinates into a sinusoidal embedding basis."""

    def __init__(self, M: int, target_dim: int, gamma: float = 1.0) -> None:
        """Initialize the random Fourier projection layer.

        Args:
            M: Input coordinate dimensionality.
            target_dim: Size of the produced embedding vector.
            gamma: Scale factor controlling the projection initialization variance.
        """
        super().__init__()
        self.gamma = gamma
        self.Wr = nn.Linear(M, target_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates into concatenated cosine and sine features.

        Args:
            x: Coordinate tensor whose last dimension matches ``M``.

        Returns:
            A tensor whose last dimension is the configured positional embedding size.
        """
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.cat([cosines, sines], dim=-1)
        return emb


class FourierEncoding(nn.Module):
    def __init__(
        self,
        target_dim: int,
        max_freq = 1,
        freq_init="uniform",
        init_mode="random",
        include_raw:bool=True,
        learnable_freqs:bool=True,
        device='cuda'
    ) -> None:
        """Initialize a learnable 1D Fourier encoding module.

        Args:
            target_dim: Width of the generated encoding vector.
            max_freq: Highest initial frequency magnitude before scaling by ``2π``.
            freq_init: Distribution family for the initialized frequencies (``uniform`` or ``log``).
            init_mode: Whether to sample frequencies randomly or place them on a linear grid.
            include_raw: Whether to prepend the raw coordinate to the embedding.
            learnable_freqs: Whether the frequency coefficients should be trainable.
            device: Device on which to allocate the frequency parameters.
        """
        super().__init__()
        self.target_dim = target_dim
        freq_dim = ((target_dim - int(include_raw)) // 2) + 1  # +1 for padding with odd target_dim
        adj_max_freq = float(max_freq * 2 * torch.pi)

        if init_mode == "random":
            base = torch.rand(freq_dim, device=device)
        elif init_mode == "linear":
            base = torch.linspace(0.0, 1.0, steps=freq_dim, device=device)
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")

        if freq_init == "uniform":
            freqs = base * adj_max_freq
        elif freq_init == "log":
            if adj_max_freq <= 0.0:
                raise ValueError("max_freq must be positive when freq_init='log'")
            log_min = min(0.0, math.log(adj_max_freq))
            log_max = max(0.0, math.log(adj_max_freq))
            freqs = torch.exp(log_min + base * (log_max - log_min))
        else:
            raise ValueError(f"Unsupported freq_init: {freq_init}")

        self.freqs = nn.Parameter(freqs, requires_grad=learnable_freqs)
        self.include_raw = include_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar positions with sinusoidal features.

        Args:
            x: Input coordinate tensor to encode.

        Returns:
            A contiguous tensor truncated to ``target_dim`` features per position.
        """
        x = x.unsqueeze(-1)  # (N, M, 1)
        projected = x * self.freqs
        cosines, sines = torch.cos(projected), torch.sin(projected)
        if self.include_raw:
            emb = torch.cat([x, cosines, sines], dim=-1)
        else:
            emb = torch.cat([cosines, sines], dim=-1)
        trunc_emb = emb[..., :self.target_dim]
        if trunc_emb.dim() > 2 and trunc_emb.shape[1] == 1:
            trunc_emb = trunc_emb.squeeze(1)
        return trunc_emb.contiguous()
