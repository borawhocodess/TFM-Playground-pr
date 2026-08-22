"""Compatibility imports for the former singular prior module."""

from tfmplayground.priors import (
    MAX_NUM_CLASSES,
    Batch,
    DictPrior,
    DumpPrior,
    FunctionPrior,
    ModdedNanoPrior,
    NanoTabICLPrior,
    Prior,
    PriorDataLoader,
    SCMPrior,
    dump_prior_to_h5,
    get_batch,
)
from tfmplayground.priors import scm as _scm


def __getattr__(name: str):
    """Forward former SCM implementation details while callers migrate."""
    try:
        return getattr(_scm, name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error


__all__ = [
    "MAX_NUM_CLASSES",
    "Batch",
    "DictPrior",
    "DumpPrior",
    "FunctionPrior",
    "ModdedNanoPrior",
    "NanoTabICLPrior",
    "Prior",
    "PriorDataLoader",
    "SCMPrior",
    "dump_prior_to_h5",
    "get_batch",
]
