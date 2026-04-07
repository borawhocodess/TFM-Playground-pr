"""Quantile (pinball) loss for regression — replaces bucket CE."""
import torch
from torch import nn


class QuantileLoss(nn.Module):
    """
    Pinball loss over N_QUANTILES evenly-spaced quantile levels.

    pred:    (..., N_QUANTILES) — raw model outputs (no activation needed)
    targets: (...)              — z-scored regression targets

    Returns per-sample loss (...) — sum of pinball losses over all quantiles.
    Call .mean() on the result as usual.
    """
    N_QUANTILES = 999

    def __init__(self):
        super().__init__()
        alphas = torch.linspace(1 / 1000, 999 / 1000, self.N_QUANTILES)
        self.register_buffer("alphas", alphas)

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pred: (..., 999), targets: (...)
        alphas  = self.alphas                           # (999,)
        errors  = targets.unsqueeze(-1) - pred          # (..., 999)
        loss    = torch.where(
            errors >= 0,
            alphas * errors,
            (alphas - 1.0) * errors,
        )
        return loss.mean(dim=-1)                         # (...) mean over quantiles
