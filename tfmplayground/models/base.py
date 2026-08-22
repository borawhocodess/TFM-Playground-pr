from abc import ABC, abstractmethod

import torch
from torch import nn

OUTPUT_KINDS = ("bar", "quantiles", "scalar")
PROBLEMS = ("classification", "regression")


class TabularFoundationModel(nn.Module, ABC):
    """
    Maps (X_train, y_train, X_test) to per test row outputs, and says what those outputs mean.

    The shape of a regression head does not tell you its meaning, and the prior cannot tell you
    either, so a model states it. The upstream repos disagree on this: the tabpfn lineage emits
    bucket logits for a bar distribution, tabicl and nanotabicl emit quantiles, and tabdpt and
    tabfm emit one scalar trained under mse. Getting it wrong trains against the wrong objective
    without failing, so it is declared rather than guessed.

    Attributes:
        output_kind (str): what regression outputs are. "bar" for bucket logits under a
            FullSupportBarDistribution, "quantiles" for pinball loss, "scalar" for a single point
            prediction, which nothing here trains yet.
        problems (tuple): the problems this model can be built for. A model whose y encoder is a
            class lookup cannot take continuous targets, and says so by leaving regression out.
    """

    output_kind: str = "bar"
    problems: tuple[str, ...] = PROBLEMS

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor: ...
