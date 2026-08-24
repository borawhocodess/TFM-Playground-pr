from abc import ABC, abstractmethod

from torch import nn


class TabularFoundationModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, X_train, y_train, X_test): ...
