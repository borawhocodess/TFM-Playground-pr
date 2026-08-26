import inspect
from dataclasses import asdict, fields

import pytest

from tfmplayground.configs.models import (
    ModdedNanoTabPFNClassifierConfig,
    ModdedNanoTabPFNRegressorConfig,
    NanoTabICLClassifierConfig,
    NanoTabICLRegressorConfig,
    NanoTabPFNClassifierConfig,
    NanoTabPFNRegressorConfig,
    TabFMClassifierConfig,
    TabFMRegressorConfig,
    TabICLClassifierConfig,
    TabICLRegressorConfig,
)
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFNModel
from tfmplayground.models.nanotabicl import NanoTabICLModel
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.models.tabicl import TabICLModel

PAIRS = [
    (NanoTabPFNModel, NanoTabPFNClassifierConfig),
    (NanoTabPFNModel, NanoTabPFNRegressorConfig),
    (TabICLModel, TabICLClassifierConfig),
    (TabICLModel, TabICLRegressorConfig),
    (NanoTabICLModel, NanoTabICLClassifierConfig),
    (NanoTabICLModel, NanoTabICLRegressorConfig),
    (ModdedNanoTabPFNModel, ModdedNanoTabPFNClassifierConfig),
    (ModdedNanoTabPFNModel, ModdedNanoTabPFNRegressorConfig),
    (TabFMModel, TabFMClassifierConfig),
    (TabFMModel, TabFMRegressorConfig),
]


def constructor_parameters(model_class):
    # an adapter may take **kwargs and pass them to the vendored class, so walk the chain
    names = set()
    for cls in model_class.__mro__:
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        parameters = inspect.signature(init).parameters
        names |= {name for name in parameters if name not in ("self", "args", "kwargs")}
    return names


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_config_names_match_the_constructor(model_class, config_class):
    ours = {field.name for field in fields(config_class)}
    theirs = constructor_parameters(model_class)
    assert ours == theirs


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_config_builds_its_model(model_class, config_class):
    model = model_class(**asdict(config_class()))
    assert any(parameter.requires_grad for parameter in model.parameters())
