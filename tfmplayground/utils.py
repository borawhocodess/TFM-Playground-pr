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
    """Read real targets and normalize them from each table's training split."""
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, stored_num_datapoints = y.shape
        lengths = f.get("num_datapoints")
        split_key = "train_test_split_index" if "train_test_split_index" in f else "single_eval_pos"
        splits = f[split_key]
        collected = []
        total = 0

        for table_index in range(num_tables):
            length = int(lengths[table_index]) if lengths is not None else stored_num_datapoints
            split = int(splits[table_index])
            if not 1 < split <= length:
                raise ValueError(
                    f"table {table_index} has invalid training split {split} for {length} datapoints"
                )

            values = np.asarray(y[table_index, :length], dtype=np.float32)
            train_values = values[:split]
            mean = train_values.mean()
            std = train_values.std(ddof=1) + 1e-8
            normalized = (values - mean) / std

            remaining = max_y - total
            if remaining <= 0:
                break
            collected.append(normalized[:remaining])
            total += min(normalized.size, remaining)

    return np.concatenate(collected) if collected else np.array([], dtype=np.float32)


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
