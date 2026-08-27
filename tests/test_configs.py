import inspect
import re
from dataclasses import fields
from types import SimpleNamespace

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
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ExperimentConfig,
    RegressionExperimentConfig,
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
        names |= {name for name in parameters if name not in ("self", "args", "kwargs", "config")}
    return names


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_config_covers_every_constructor_parameter(model_class, config_class):
    ours = {field.name for field in fields(config_class)}
    theirs = constructor_parameters(model_class)
    assert theirs <= ours


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_config_builds_its_model(model_class, config_class):
    model = model_class(config=config_class())
    assert any(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_every_config_field_reaches_the_model(model_class, config_class):
    source = inspect.getsource(model_class.__init__)
    read = set(re.findall(r"config\.(\w+)", source))
    ours = {field.name for field in fields(config_class)}
    unread = ours - read
    assert unread <= {"head", "problem"}
    assert read <= ours


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_the_model_keeps_its_config(model_class, config_class):
    config = config_class()
    assert model_class(config=config).config is config


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_config_may_be_any_object_with_the_fields(model_class, config_class):
    # the adapters read attributes, so a config need not be one of our dataclasses
    plain = SimpleNamespace(**{f.name: getattr(config_class(), f.name) for f in fields(config_class)})
    model = model_class(config=plain)
    assert any(parameter.requires_grad for parameter in model.parameters())


def test_the_experiment_base_cannot_name_a_problem():
    assert not hasattr(ExperimentConfig(), "problem")


def test_each_experiment_config_names_its_own_problem():
    assert ClassificationExperimentConfig().problem == "classification"
    assert RegressionExperimentConfig().problem == "regression"


def test_every_experiment_config_shares_the_defaults():
    for config in (ClassificationExperimentConfig(), RegressionExperimentConfig()):
        assert config.name == "test"
        assert config.experiments_dir == "workdir/experiments"


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_every_model_config_names_its_problem(model_class, config_class):
    name = config_class.__name__
    expected = "classification" if "Classifier" in name else "regression"
    assert config_class().problem == expected


@pytest.mark.parametrize(("model_class", "config_class"), PAIRS, ids=lambda p: p.__name__)
def test_only_a_regression_config_names_a_head(model_class, config_class):
    config = config_class()
    assert hasattr(config, "head") == (config.problem == "regression")
