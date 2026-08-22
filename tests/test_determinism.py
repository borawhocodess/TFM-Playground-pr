import numpy as np
import pytest
import torch
from torch import nn

from tfmplayground.interface import TabularClassifier, TabularRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def test_classifier_is_deterministic():
    """A randomly initialized classifier should give identical outputs for identical inputs."""
    model = NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=2,
        mlp_hidden_size=32,
        num_layers=2,
        num_outputs=3,
    )
    classifier = TabularClassifier(model, device="cpu")

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 4))
    y_train = rng.integers(0, 3, size=20)
    X_test = rng.standard_normal((10, 4))

    classifier.fit(X_train, y_train)
    first = classifier.predict_proba(X_test)
    second = classifier.predict_proba(X_test)

    np.testing.assert_array_equal(first, second)


class FixedOutputModel(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_outputs))
        self.training_states = []
        self.seen_y = None

    def forward(self, X_train, y_train, X_test):
        self.training_states.append(self.training)
        self.seen_y = y_train.detach().cpu()
        return self.bias.expand(X_test.shape[0], X_test.shape[1], -1)


def test_classifier_encodes_and_restores_non_contiguous_labels():
    model = FixedOutputModel(num_outputs=2)
    classifier = TabularClassifier(model, device="cpu")
    X_train = np.arange(12, dtype=float).reshape(6, 2)
    y_train = np.array(["zebra", "ant", "zebra", "ant", "zebra", "ant"])

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(np.ones((2, 2)))

    assert predictions.tolist() == ["ant", "ant"]
    assert set(model.seen_y.flatten().tolist()) == {0.0, 1.0}
    assert model.training_states == [False]
    assert model.training


def test_regressor_decodes_a_one_output_model_without_a_distribution():
    regressor = TabularRegressor(FixedOutputModel(num_outputs=1), device="cpu")
    regressor.fit(np.arange(12, dtype=float).reshape(6, 2), np.arange(6, dtype=float))

    assert regressor.predict(np.ones((2, 2))).shape == (2,)


def test_regressor_rejects_a_multi_output_model_without_a_decoder():
    regressor = TabularRegressor(FixedOutputModel(num_outputs=3), device="cpu")
    regressor.fit(np.arange(12, dtype=float).reshape(6, 2), np.arange(6, dtype=float))

    with pytest.raises(ValueError, match="decode a multi-output model"):
        regressor.predict(np.ones((2, 2)))
