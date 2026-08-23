from abc import ABC, abstractmethod

import torch
from torch import nn

OUTPUT_KINDS = ("class_logits", "bar", "fixed_bin_logits", "quantiles", "scalar")
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
        output_kind (str): what this instance's head emits. "class_logits" when it was built for
            classification. Otherwise, for regression: "bar" for bucket logits whose edges are
            fitted from the prior, "fixed_bin_logits" for bucket logits over edges the model
            itself fixes, "quantiles" for pinball loss, and "scalar" for a single point
            prediction, which nothing here trains yet. A model declaring "fixed_bin_logits" must
            also provide regression_borders(), since fitting edges from the prior would hand its
            channels ranges it never meant.
            A model that serves both problems off one generic head, like the tabpfn lineage,
            keeps its regression kind and leans on problems instead.
        problems (tuple): the problems this model can be built for. A model whose y encoder is a
            class lookup cannot take continuous targets, and says so by leaving regression out.
        num_outputs (int | None): outputs per test row, when the model knows. None means it will
            be probed with a throwaway forward pass, which is the fallback for third party models.
    """

    output_kind: str = "bar"
    problems: tuple[str, ...] = PROBLEMS
    num_outputs: int | None = None

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor: ...
