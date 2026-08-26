from abc import ABC, abstractmethod


class Prior(ABC):
    @abstractmethod
    def batch(self, batch_size): ...


class PriorDataLoader:
    def __init__(self, prior, batch_size):
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.prior.batch(self.batch_size)
