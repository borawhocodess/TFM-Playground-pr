from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import torch


class Prior(ABC):
    """
    base class for every prior this package trains on
    """

    config = None

    @abstractmethod
    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gives one batch of tables, split into train and test parts
        """
        ...


class PriorDataLoader:
    """
    turns prior into endless iterator of batches
    """

    def __init__(self, prior: Prior, batch_size: int) -> None:
        """
        keeps prior and batch size
        """
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        gives batches from prior without end
        """
        while True:
            yield self.prior.batch(self.batch_size)


class DictPrior(Prior):
    """
    adapts loaders that give dict batches to prior interface
    """

    def __init__(self, loader: Iterable[dict]) -> None:
        """
        keeps loader and starts its iterator
        """
        self.loader = loader
        self.batches = iter(loader)
        self.config = getattr(loader, "config", None)

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gives next dict batch from loader, split at train test index

        starts loader again at its end, so batches never run out
        """
        try:
            d = next(self.batches)
        except StopIteration:
            self.batches = iter(self.loader)
            d = next(self.batches)
        x = d["x"]
        y = d["y"]
        target_y = d["target_y"]
        sep = int(d["train_test_split_index"])
        loaded_batch_size = x.shape[0]
        if loaded_batch_size != batch_size:
            raise ValueError(f"batch size is {batch_size} but loader gives {loaded_batch_size}")
        return x[:, :sep], y[:, :sep], x[:, sep:], target_y[:, sep:]
