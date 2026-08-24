from contextlib import contextmanager, nullcontext

import torch
from tabicl._model.tabicl import TabICL
from torch import nn

from tfmplayground.models.base import TabularFoundationModel


class TabICLModel(TabICL, TabularFoundationModel):
    def forward(self, X_train, y_train, X_test):
        X = torch.cat([X_train, X_test], dim=1)
        with nullcontext() if self.training else self.train_path_without_dropout():
            return super().forward(X, y_train)

    @contextmanager
    def train_path_without_dropout(self):
        if not hasattr(self, "_dropout_module_state"):
            dropout_modules = [module for module in self.modules() if isinstance(module, nn.Dropout)]
            rate_modules = [
                (module, module.dropout)
                for module in self.modules()
                if isinstance(getattr(module, "dropout", None), float | int)
            ]
            self._dropout_module_state = (dropout_modules, rate_modules)
        dropout_modules, rate_modules = self._dropout_module_state
        self.train()
        for module in dropout_modules:
            module.train(False)
        for module, _ in rate_modules:
            module.dropout = 0.0
        try:
            yield
        finally:
            for module, rate in rate_modules:
                module.dropout = rate
            self.eval()
