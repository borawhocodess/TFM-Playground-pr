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


class FixedBinLoss(nn.Module):
    """
    Cross entropy over bins the model fixed, decoded the way tabdpt v1.2 decodes.

    The decoder is upstream's exactly: softmax the bin logits, take the bin centres, sum them
    weighted. The objective is not, because tabdpt's v1.2 training code is not public and the
    older training repo regressed a single scalar under mse instead. Cross entropy on the bin a
    target falls into is the ordinary choice for a binned head, and it is ours, not theirs.

    Unlike a FullSupportBarDistribution the outer bins carry no tails, so targets outside the
    range fall into the end bins rather than into a half normal. That is what fixed bins mean.

    Args:
        borders (torch.Tensor): n_bins + 1 fixed edges, from the model that owns them.
    """

    def __init__(self, borders: torch.Tensor):
        super().__init__()
        self.register_buffer("borders", borders)
        self.register_buffer("centers", (borders[:-1] + borders[1:]) / 2)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        borders = self.borders.to(logits.device)
        index = torch.bucketize(target.detach(), borders[1:-1])
        losses = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), index.reshape(-1), reduction="none")
        return losses.reshape(target.shape)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1) @ self.centers.to(logits.device)


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
