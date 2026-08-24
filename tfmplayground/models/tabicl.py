import torch
from tabicl._model.tabicl import TabICL

from tfmplayground.models.base import TabularFoundationModel


class TabICLModel(TabICL, TabularFoundationModel):
    def forward(self, X_train, y_train, X_test):
        X = torch.cat([X_train, X_test], dim=1)
        mode = self.training
        self.train()  # tabicl selects another device in eval mode
        try:
            return super().forward(X, y_train)
        finally:
            self.train(mode)
