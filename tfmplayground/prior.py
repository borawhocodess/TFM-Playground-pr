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
