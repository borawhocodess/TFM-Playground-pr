import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from pfns.bar_distribution import get_bucket_limits
from torch import nn

CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(name):
    """Load a YAML config from tfmplayground/configs."""
    path = Path(name)
    if not path.exists():
        path = CONFIGS_DIR / f"{Path(name).stem}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


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


class QuantileLoss(nn.Module):
    """Pinball loss summed over a fixed grid of quantile levels."""

    def __init__(self, n_quantiles: int):
        super().__init__()
        alphas = torch.arange(1, n_quantiles + 1, dtype=torch.float) / (n_quantiles + 1)
        self.register_buffer("alphas", alphas)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alphas = self.alphas.to(logits.device)
        error = target.unsqueeze(-1) - logits
        losses = torch.maximum(alphas * error, (alphas - 1.0) * error)
        return losses.sum(dim=-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.mean(dim=-1)
