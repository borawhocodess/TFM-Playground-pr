from abc import ABC, abstractmethod

import torch
from torch import nn


class TabularFoundationModel(nn.Module, ABC):
    """Common in-context-learning contract for tabular foundation models.

    Inputs use batch-first shapes: ``X_train`` is ``(batch, train_rows, features)``,
    ``y_train`` is ``(batch, train_rows)``, and ``X_test`` is
    ``(batch, test_rows, features)``. Implementations return
    ``(batch, test_rows, outputs)`` and must derive preprocessing statistics only
    from the training context.

    Classification outputs are unnormalized class logits. Regression output
    semantics depend on the matching training criterion and decoder; a shared
    shape does not make every model/criterion combination interchangeable.
    """

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Predict outputs for ``X_test`` using ``X_train`` and ``y_train`` as context."""
        ...
