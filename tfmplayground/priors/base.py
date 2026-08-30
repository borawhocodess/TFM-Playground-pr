from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import torch


class Prior(ABC):
    config = None

    @abstractmethod
    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        todo
        """
        ...


class PriorDataLoader:
    def __init__(self, prior: Prior, batch_size: int) -> None:
        """
        todo
        """
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        todo
        """
        while True:
            yield self.prior.batch(self.batch_size)


class DictPrior(Prior):
    def __init__(self, loader: Iterable[dict]) -> None:
        """
        todo
        """
        self.loader = loader
        self.batches = iter(loader)
        self.config = getattr(loader, "config", None)

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        todo
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
