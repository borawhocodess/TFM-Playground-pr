import pytest
import torch

from tfmplayground.models import (
    ModdedNanoTabPFNModel,
    NanoTabICLModel,
    NanoTabPFNModel,
    TabFMModel,
    TabICLModel,
)


def make_nanotabpfn():
    return NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=2,
        mlp_hidden_size=32,
        num_layers=2,
        num_outputs=3,
    )


def make_nanotabicl():
    return NanoTabICLModel(
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


def make_moddednanotabpfn():
    return ModdedNanoTabPFNModel(l=2, a=2, e=16, h=32, o=3)


def make_tabfm():
    return TabFMModel(max_classes=3, is_classifier=True)


def make_tabicl():
    return TabICLModel(
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
