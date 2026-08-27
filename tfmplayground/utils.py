import random
import uuid
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from pfns.bar_distribution import get_bucket_limits
from torch import nn


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def make_bucket_borders(prior, num_buckets, batch_size, min_targets, outlier_threshold):
    normalized_targets = []
    collected = 0
    while collected < min_targets:
        _, y_train, _, y_test = prior.batch(batch_size)
        y_train = y_train.detach().to("cpu", torch.float32)
        y_test = y_test.detach().to("cpu", torch.float32)
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True) + 1e-8
        y = torch.cat([y_train, y_test], dim=1)
        normalized = ((y - y_mean) / y_std).ravel()
        if normalized.numel() == 0:
            raise ValueError("the prior gives no targets")
        normalized_targets.append(normalized)
        collected += normalized.numel()

    ys = torch.cat(normalized_targets)
    ys = ys[torch.isfinite(ys)]
    if ys.numel() < num_buckets:
        raise ValueError(f"{ys.numel()} targets cannot make {num_buckets} buckets")

    mean = ys.mean()
    std = ys.std()
    inside = ys[(ys - mean).abs() <= outlier_threshold * std]
    robust_mean = inside.mean()
    cut_off = outlier_threshold * inside.std()
    ys = ys.clamp(robust_mean - cut_off, robust_mean + cut_off)

    n = (ys.numel() // num_buckets) * num_buckets
    ys = ys[:n]
    ys_per_bucket = n // num_buckets
    ys_sorted, _ = torch.sort(ys)
    chunks = ys_sorted.reshape(num_buckets, ys_per_bucket)
    interiors = (chunks[:-1, -1] + chunks[1:, 0]) / 2
    min_outer = ys_sorted[0].unsqueeze(0)
    max_outer = ys_sorted[-1].unsqueeze(0)
    borders = torch.cat([min_outer, interiors, max_outer])

    if borders.numel() != num_buckets + 1:
        raise ValueError(f"{borders.numel()} borders cannot make {num_buckets} buckets")
    if borders.numel() != torch.unique_consecutive(borders).numel():
        raise ValueError("the targets repeat, so one bucket has no width")
    return borders


def make_global_bucket_edges(filename, n_buckets=100, device=None, max_y=5_000_000):
    if device is None:
        device = get_default_device()
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, num_datapoints = y.shape

        num_tables_to_use = min(num_tables, max_y // num_datapoints)

        y_subset = np.array(y[:num_tables_to_use, :], dtype=np.float32)
        y_means = y_subset.mean(axis=1, keepdims=True)
        y_stds = y_subset.std(axis=1, keepdims=True, ddof=1) + 1e-8
        ys_concat = ((y_subset - y_means) / y_stds).ravel()

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges


class ScalarMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, logits, target):
        scalars = logits.reshape(target.shape)
        return super().forward(scalars, target)

    def mean(self, logits):
        return logits.squeeze(-1)


class QuantileLoss(nn.Module):
    """Pinball loss averaged over a fixed grid of quantile levels."""

    def __init__(self, n_quantiles: int):
        super().__init__()
        alphas = torch.arange(1, n_quantiles + 1, dtype=torch.float) / (n_quantiles + 1)
        self.register_buffer("alphas", alphas)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alphas = self.alphas.to(logits.device)
        error = target.unsqueeze(-1) - logits
        losses = torch.maximum(alphas * error, (alphas - 1.0) * error)
        return losses.mean(dim=-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.mean(dim=-1)


class Experiment:
    def __init__(self, config):
        self.started = datetime.now()
        timestamp = self.started.strftime("%y%m%d-%H%M%S")
        uid = uuid.uuid4().hex[:8]
        name = config.name.strip()
        self.id = f"{timestamp}-{uid}-{name}" if name else f"{timestamp}-{uid}"
        self.dir = Path(config.experiments_dir) / config.problem / name / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / f"{self.id}-log.txt"

    def print0(self, s, console=False):
        with open(self.log_path, "a") as f:
            if console:
                print(s)
            print(s, file=f)
