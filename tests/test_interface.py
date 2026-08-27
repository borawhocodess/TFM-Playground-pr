import numpy as np
import pytest
import torch
from pfns.bar_distribution import FullSupportBarDistribution

from tfmplayground.configs.models import NanoTabPFNClassifierConfig, NanoTabPFNRegressorConfig
from tfmplayground.interface import TabularClassifier, TabularRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import QuantileLoss, ScalarMSELoss


def classifier(outputs=10):
    config = NanoTabPFNClassifierConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=outputs
    )
    return TabularClassifier(NanoTabPFNModel(config=config), device="cpu")


def table(rows=20, features=4, seed=0):
    return np.random.default_rng(seed).standard_normal((rows, features))


def test_string_labels_are_accepted():
    model = classifier()
    labels = np.array(["cat", "dog"] * 10)
    model.fit(table(), labels)
    assert set(model.predict(table(5, seed=1))) <= {"cat", "dog"}


def test_labels_that_skip_numbers_do_not_widen_the_head():
    model = classifier()
    labels = np.array([0, 5, 9] * 6 + [0, 5])
    model.fit(table(), labels)
    assert model.num_classes == 3
    assert model.predict_proba(table(5, seed=1)).shape == (5, 3)


def test_labels_that_skip_numbers_come_back_as_themselves():
    model = classifier()
    labels = np.array([0, 5, 9] * 6 + [0, 5])
    model.fit(table(), labels)
    assert set(model.predict(table(5, seed=1))) <= {0, 5, 9}


def test_labels_that_start_at_one_carry_no_phantom_class():
    model = classifier()
    labels = np.array([1, 2, 3] * 6 + [1, 2])
    model.fit(table(), labels)
    assert model.num_classes == 3


def test_contiguous_labels_are_unchanged():
    model = classifier()
    labels = np.array([0, 1, 2] * 6 + [0, 1])
    model.fit(table(), labels)
    assert model.num_classes == 3
    assert set(model.predict(table(5, seed=1))) <= {0, 1, 2}


def test_fit_hands_back_the_classifier():
    model = classifier()
    assert model.fit(table(), np.array([0, 1] * 10)) is model


def test_fit_hands_back_the_regressor():
    config = NanoTabPFNRegressorConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=9
    )
    config.head = "quantiles"
    model = TabularRegressor(NanoTabPFNModel(config=config), device="cpu")
    assert model.fit(table(), np.arange(20, dtype=float)) is model


def test_the_regressor_builds_the_decoder_the_model_declares():
    config = NanoTabPFNRegressorConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=9
    )
    config.head = "quantiles"
    regressor = TabularRegressor(NanoTabPFNModel(config=config), device="cpu")
    assert isinstance(regressor.dist, QuantileLoss)
    assert regressor.dist.alphas.numel() == 9


def test_a_scalar_model_needs_no_borders_to_decode():
    config = NanoTabPFNRegressorConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=1
    )
    config.head = "scalar"
    assert isinstance(TabularRegressor(NanoTabPFNModel(config=config), device="cpu").dist, ScalarMSELoss)


def test_flat_borders_cannot_decode_buckets():
    config = NanoTabPFNRegressorConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=9
    )
    with pytest.raises(ValueError, match="borders are flat"):
        TabularRegressor(NanoTabPFNModel(config=config), device="cpu")


def test_fitted_borders_decode_buckets():
    config = NanoTabPFNRegressorConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=9
    )
    model = NanoTabPFNModel(config=config)
    model.borders = torch.linspace(-3, 3, 10)
    assert isinstance(TabularRegressor(model, device="cpu").dist, FullSupportBarDistribution)


def test_the_probabilities_sum_to_one():
    model = classifier()
    model.fit(table(), np.array([0, 1] * 10))
    assert np.allclose(model.predict_proba(table(5, seed=1)).sum(axis=1), 1.0)
