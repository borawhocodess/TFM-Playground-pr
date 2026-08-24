from contextlib import contextmanager

import torch
from tabicl._model.tabicl import TabICL
from torch import nn

from tfmplayground.models.base import TabularFoundationModel


class TabICLModel(TabICL, TabularFoundationModel):
    def forward(self, X_train, y_train, X_test):
        X = torch.cat([X_train, X_test], dim=1)
        if self.training:
            return super().forward(X, y_train)
        with self.train_path_without_dropout():
            return super().forward(X, y_train)

    @contextmanager
    def train_path_without_dropout(self):
        rates = []
        self.train()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train(False)
            rate = getattr(module, "dropout", None)
            if isinstance(rate, float | int):
                rates.append((module, rate))
                module.dropout = 0.0
        try:
            yield
        finally:
            for module, rate in rates:
                module.dropout = rate
            self.eval()
