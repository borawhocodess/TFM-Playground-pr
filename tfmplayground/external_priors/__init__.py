"""Interfaces to external prior libraries (TabICL, TICL, TabPFN v1)."""

import importlib

# imported on first use rather than up front, so one library failing to import does not take the
# others down with it. ticl imports interpret without declaring it as a dependency, for one.
SOURCES = {
    "TabICLPrior": "tabicl",
    "TabICLPriorDataLoader": "tabicl",
    "TICLPriorDataLoader": "ticl",
    "build_ticl_prior": "ticl",
    "TabPFNPriorDataLoader": "tabpfn",
    "build_tabpfn_prior": "tabpfn",
}

__all__ = sorted(SOURCES)


def __getattr__(name: str):
    if name not in SOURCES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{SOURCES[name]}", __name__), name)


def __dir__() -> list[str]:
    return __all__
