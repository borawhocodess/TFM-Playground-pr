from abc import ABC, abstractmethod

import torch
from torch import nn

OUTPUT_KINDS = ("bar_logits", "class_logits", "generic_logits", "quantiles", "scalar")
PROBLEMS = ("classification", "regression")


class TabularFoundationModel(nn.Module, ABC):
    """
    Maps (X_train, y_train, X_test) to per test row outputs, and says what those outputs mean.

    The shape of a head does not tell you its meaning, and the prior cannot tell you either, so a
    model states it for each supported problem. Getting it wrong can train against the wrong
    objective without failing.

    Attributes:
        output_kinds (dict): maps each supported problem to class logits, fitted bar logits,
            generic logits, quantiles, or one scalar. "generic_logits" is the
            tabpfn lineage head: n logits whose meaning the criterion decides, so it defaults to a
            fitted bar distribution and also accepts a quantile loss, while the structural kinds
            accept exactly one criterion because the channels have to be read as they were written.
        num_outputs (int): number of values returned for each test row.
    """

    output_kinds: dict[str, str] = {
        "classification": "class_logits",
        "regression": "generic_logits",
    }
    problems: tuple[str, ...] = PROBLEMS
    num_outputs: int | None = None

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor: ...
