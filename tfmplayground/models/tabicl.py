import torch
from tabicl._model.tabicl import TabICL

from tfmplayground.models.base import TabularFoundationModel


class TabICLModel(TabICL, TabularFoundationModel):
    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        X = torch.cat([X_train, X_test], dim=1)
        mode = self.training
        self.train()  # tabicl selects another device in eval mode
        try:
            return super().forward(X, y_train)
        finally:
            self.train(mode)
