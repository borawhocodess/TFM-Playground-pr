import numpy as np
import pytest
import torch

from tfmplayground.configs.models import (
    ModdedNanoTabPFNClassifierConfig,
    NanoTabICLClassifierConfig,
    NanoTabPFNClassifierConfig,
    TabFMClassifierConfig,
    TabICLClassifierConfig,
)
from tfmplayground.interface import TabularClassifier
from tfmplayground.models import (
    ModdedNanoTabPFNModel,
    NanoTabICLModel,
    NanoTabPFNModel,
    TabFMModel,
    TabICLModel,
)


def make_nanotabpfn():
    return NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16,
            num_attention_heads=2,
            mlp_hidden_size=32,
            num_layers=2,
            num_outputs=3,
        )
    )


def make_nanotabicl():
    return NanoTabICLModel(
        config=NanoTabICLClassifierConfig(
            max_classes=3,
            out_dim=3,
            embed_dim=16,
            col_num_blocks=1,
            row_num_blocks=2,
            icl_num_blocks=2,
            col_nhead=2,
            row_nhead=2,
            icl_nhead=2,
            n_cls_cols=2,
            n_cls_rows=8,
        )
    )


def make_moddednanotabpfn():
    return ModdedNanoTabPFNModel(config=ModdedNanoTabPFNClassifierConfig(l=2, a=2, e=16, h=32, o=3))


def make_tabfm():
    return TabFMModel(config=TabFMClassifierConfig(max_classes=3))


def make_tabicl():
    return TabICLModel(
        config=TabICLClassifierConfig(
            max_classes=3,
            embed_dim=32,
            col_num_blocks=1,
            row_num_blocks=1,
            icl_num_blocks=1,
            col_nhead=2,
            row_nhead=2,
            icl_nhead=2,
            col_num_inds=8,
            row_num_cls=2,
        )
    )


@pytest.mark.parametrize(
    "make_model",
    [make_nanotabpfn, make_nanotabicl, make_moddednanotabpfn, make_tabfm, make_tabicl],
    ids=["nanotabpfn", "nanotabicl", "moddednanotabpfn", "tabfm", "tabicl"],
)
def test_forward_follows_base_contract(make_model):
    torch.manual_seed(0)
    model = make_model()
    X_train = torch.randn(2, 12, 4)
    y_train = torch.randint(0, 3, (2, 12)).float()
    X_test = torch.randn(2, 5, 4)

    with torch.no_grad():
        output = model(X_train, y_train, X_test)

    assert output.shape == (2, 5, 3)
    assert torch.isfinite(output).all()


def make_tabicl_with_dropout():
    torch.manual_seed(0)
    model = TabICLModel(
        config=TabICLClassifierConfig(
            max_classes=3,
            embed_dim=32,
            col_num_blocks=1,
            row_num_blocks=1,
            icl_num_blocks=1,
            col_nhead=2,
            row_nhead=2,
            icl_nhead=2,
            col_num_inds=8,
            row_num_cls=2,
            dropout=0.2,
        )
    )
    # zero_init leaves every residual branch at zero, where dropout cannot change anything
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    return model


def test_tabicl_repeats_its_predictions_with_dropout():
    model = make_tabicl_with_dropout()
    model.eval()

    X_train = torch.randn(2, 12, 4)
    y_train = torch.randint(0, 3, (2, 12))
    X_test = torch.randn(2, 5, 4)

    with torch.no_grad():
        first = model(X_train, y_train, X_test)
        second = model(X_train, y_train, X_test)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_classifier_repeats_its_predictions_with_dropout():
    torch.manual_seed(0)
    model = make_tabicl_with_dropout()
    classifier = TabularClassifier(model, device="cpu")

    rng = np.random.default_rng(0)
    classifier.fit(rng.standard_normal((20, 4)), rng.integers(0, 3, size=20))
    X_test = rng.standard_normal((10, 4))

    np.testing.assert_array_equal(classifier.predict_proba(X_test), classifier.predict_proba(X_test))


def test_classifier_predicts_in_eval_mode():
    model = make_tabicl_with_dropout()
    classifier = TabularClassifier(model, device="cpu")

    rng = np.random.default_rng(0)
    classifier.fit(rng.standard_normal((20, 4)), rng.integers(0, 3, size=20))
    model.train()
    classifier.predict_proba(rng.standard_normal((10, 4)))

    assert not model.training
