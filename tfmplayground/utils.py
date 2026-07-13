import random
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from pfns.bar_distribution import get_bucket_limits
from torch import nn

CACHE_DIR = Path.home() / ".cache" / "tfmplayground"


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


def fetch_dump(url, cache_dir=CACHE_DIR):
    """Downloads a prior dump to the cache directory if it is not already there and returns its path."""
    cache_dir = Path(cache_dir)
    path = cache_dir / url.split("/")[-1]
    if path.exists():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} to {path}", flush=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    partial = path.with_suffix(".part")
    with open(partial, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    partial.rename(path)
    return path


def make_global_bucket_edges(dump, n_buckets=100, device=None, max_y=5_000_000):
    """Fits bucket edges on the targets of a dump, given as a path or anything with a filename attribute."""
    if device is None:
        device = get_default_device()
    with h5py.File(getattr(dump, "filename", dump), "r") as f:
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
