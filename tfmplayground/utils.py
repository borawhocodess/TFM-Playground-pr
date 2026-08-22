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


def make_global_bucket_edges(prior, n_buckets=100, device=None, max_y=5_000_000, batch_size=8, max_batches=25):
    """
    Fits bucket edges on the targets of a prior, given as a path to a dump, anything carrying a
    filename attribute, or any prior we can sample batches off.
    """
    if device is None:
        device = get_default_device()
    filename = getattr(prior, "filename", prior if isinstance(prior, str | Path) else None)
    if filename is not None:
        ys_concat = dump_targets(filename, max_y)
    else:
        ys_concat = sampled_targets(prior, max_y, batch_size, max_batches)

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges


def dump_targets(filename, max_y):
    """Reads the targets of a dump and z-normalizes every table over all of its real datapoints."""
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, stored_num_datapoints = y.shape
        lengths = f.get("num_datapoints")
        collected = []
        total = 0

        for table_index in range(num_tables):
            if total >= max_y:
                break
            # tables shorter than the longest one are zero-padded in the dump, those rows are not targets
            length = int(lengths[table_index]) if lengths is not None else stored_num_datapoints
            values = np.asarray(y[table_index, :length], dtype=np.float32)
            normalized = (values - values.mean()) / (values.std(ddof=1) + 1e-8)
            collected.append(normalized[: max_y - total])
            total += min(normalized.size, max_y - total)

    if not collected:
        raise ValueError(f"{filename} holds no targets to fit bucket edges on.")
    return np.concatenate(collected)


def sampled_targets(prior, max_y, batch_size=8, max_batches=25):
    """
    Samples the targets off a prior and z-normalizes every table the way the training loop does.
    Stops at max_y targets or max_batches batches, whichever comes first.
    """
    collected = []
    total = 0
    for _ in range(max_batches):
        _, y_train, _, y_test = prior.batch(batch_size)
        y_train = y_train.detach().to("cpu", torch.float32)
        y_test = y_test.detach().to("cpu", torch.float32)
        y_means = y_train.mean(dim=1, keepdim=True)
        y_stds = y_train.std(dim=1, keepdim=True) + 1e-8
        normalized = ((torch.cat([y_train, y_test], dim=1) - y_means) / y_stds).ravel().numpy()
        collected.append(normalized)
        total += normalized.size
        if total >= max_y:
            break
    if not collected:
        raise ValueError("The prior yielded no batches to fit bucket edges on.")
    return np.concatenate(collected)


class FixedBinDistribution(nn.Module):
    """CRPS loss and expectation decoder over model-defined finite bins."""

    output_kind = "fixed_bin_logits"

    def __init__(self, borders: torch.Tensor):
        super().__init__()
        borders = torch.as_tensor(borders, dtype=torch.float32)
        if borders.ndim != 1 or borders.numel() < 2:
            raise ValueError("borders must be a one-dimensional tensor with at least two values")
        if not torch.all(borders[1:] > borders[:-1]):
            raise ValueError("borders must be strictly increasing")
        self.register_buffer("borders", borders)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != self.borders.numel() - 1:
            raise ValueError(
                f"logits have {logits.shape[-1]} bins but the distribution has {self.borders.numel() - 1}"
            )
        bins = torch.bucketize(target.contiguous(), self.borders[1:-1])
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted_cdf = probabilities.cumsum(dim=-1)
        bin_indices = torch.arange(logits.shape[-1], device=logits.device)
        target_cdf = (bin_indices >= bins.unsqueeze(-1)).to(predicted_cdf.dtype)
        widths = self.borders[1:] - self.borders[:-1]
        return (widths * (predicted_cdf - target_cdf).square()).sum(dim=-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        centres = (self.borders[:-1] + self.borders[1:]) / 2
        probabilities = torch.softmax(logits.float(), dim=-1)
        return (probabilities * centres).sum(dim=-1)


class ScalarMSELoss(nn.MSELoss):
    """Elementwise MSE for a model that emits one scalar per row."""

    output_kind = "scalar"

    def __init__(self):
        super().__init__(reduction="none")


class QuantileLoss(nn.Module):
    """Pinball loss summed over a fixed grid of quantile levels."""

    output_kind = "quantiles"

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
