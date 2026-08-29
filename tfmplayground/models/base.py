from abc import ABC, abstractmethod

import torch
from torch import nn


class TabularFoundationModel(nn.Module, ABC):
    """
    base class for every model this package trains
    """

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        predicts the test rows with the train rows as context
        """
        ...
