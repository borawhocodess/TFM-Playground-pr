from abc import ABC, abstractmethod


class Callback(ABC):
    """Abstract base class for callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            epoch_time (float): Time of the epoch in seconds.
            loss (float): Mean loss for the epoch.
            model: The model being trained.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def close(self):
        """Called to release any resources or perform cleanup."""
        pass


class Trainer(ABC):
    """Abstract base class for trainers."""

    @abstractmethod
    def train(self):
        """Run training."""
        pass
