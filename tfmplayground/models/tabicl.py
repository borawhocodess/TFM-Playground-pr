"""The official TabICL model, wrapped to take the base forward contract."""

import torch
from tabicl._model.tabicl import TabICL

from tfmplayground.models.base import TabularFoundationModel


class TabICLModel(TabularFoundationModel):
    """
    The official tabicl model, straight out of the library, no vendored copy.

    tabicl's forward already hands back (batch, test rows, outputs) in training mode, so all
    this does is concatenate the two halves back into the single table it wants. Both problems
    come off the same architecture: max_classes above 0 gives that many class logits per test
    row, max_classes at 0 gives num_quantiles regression outputs instead.

    The model lives in tabicl._model, a private module, because the public surface is the
    sklearn estimators and those predict rather than pretrain. tabicl's own trainer imports it
    from exactly here, so it is the sanctioned path for what we are doing.

    Args:
        max_classes (int): class logits per test row, 0 for regression.
        num_quantiles (int): regression outputs per test row, only read when max_classes is 0.
        **kwargs: passed through to tabicl, see TabICL for the architecture settings.
    """

    output_kind = "quantiles"  # tabicl's own docs: at max_classes=0 it regresses by quantiles

    def __init__(self, max_classes: int = 10, num_quantiles: int = 999, **kwargs):
        super().__init__()
        self.problems = ("classification",) if max_classes > 0 else ("regression",)
        self.model = TabICL(max_classes=max_classes, num_quantiles=num_quantiles, **kwargs)

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        # tabicl branches on self.training at every level, and its eval branch is the optimised
        # inference path behind the sklearn estimators: it resolves its own device and asserts
        # every table in the batch carries the same number of classes. neither holds here, so we
        # always take the training path, the one tabicl's own trainer runs. dropout is put back
        # under our own mode afterwards so predicting through a trained model stays deterministic.
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train(self.training)
        try:
            return self.model(torch.cat([X_train, X_test], dim=1), y_train)
        finally:
            for module, was_training in modes:
                module.training = was_training
