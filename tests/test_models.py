import pytest
import torch
from torch import nn

from tfmplayground import pretrainTFM
from tfmplayground.prior import PriorDataLoader
from tfmplayground.models import (
    ModdedNanoTabPFNModel,
    NanoTabICLv2,
    NanoTabPFNModel,
    TabDPTModel,
    TabFMModel,
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
    return NanoTabICLv2(
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


def make_nanotabdpt():
    return TabDPTModel(
        dropout=0.0,
        enc_cell_dim=8,
        n_out=3,
        regression_bin_count=8,
        regression_bin_min=-3.0,
        regression_bin_max=3.0,
        nhead=2,
        nhid=32,
        ninp=16,
        nlayers=2,
        num_features=8,
        base_len=32,
        max_len=64,
        y_encoder_dim=8,
        classification=True,
    )


def make_moddednanotabpfn():
    return ModdedNanoTabPFNModel(l=2, a=2, e=16, h=32, o=3)


def make_nanotabfm():
    return TabFMModel(max_classes=3, is_classifier=True)


@pytest.mark.parametrize(
    "make_model",
    [make_nanotabpfn, make_nanotabicl, make_nanotabdpt, make_moddednanotabpfn, make_nanotabfm],
    ids=["nanotabpfn", "nanotabicl", "nanotabdpt", "moddednanotabpfn", "nanotabfm"],
)
def test_forward_follows_base_contract(make_model):
    """Every model maps (X_train, y_train, X_test) to per-test-row logits."""
    torch.manual_seed(0)
    model = make_model()
    X_train = torch.randn(2, 12, 4)
    y_train = torch.randint(0, 3, (2, 12)).float()
    X_test = torch.randn(2, 5, 4)

    with torch.no_grad():
        out = model(X_train, y_train, X_test)

    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()


def get_three_class_batch(batch_size, num_datapoints, num_features):
    x = torch.randn(batch_size, num_datapoints, num_features)
    y = (x[:, :, 0] > 0).float() + (x[:, :, 1] > 0).float()
    return dict(x=x, y=y, target_y=y, train_test_split_index=num_datapoints // 2)


@pytest.mark.parametrize(
    "make_model",
    [make_nanotabpfn, make_nanotabicl, make_nanotabdpt, make_moddednanotabpfn, make_nanotabfm],
    ids=["nanotabpfn", "nanotabicl", "nanotabdpt", "moddednanotabpfn", "nanotabfm"],
)
def test_pretrainTFM_swaps_any_model(make_model):
    """Every model drops into pretrainTFM as-is and comes back trained."""
    torch.manual_seed(0)
    prior = PriorDataLoader(
        get_batch_function=get_three_class_batch,
        num_steps=2,
        batch_size=2,
        num_datapoints_max=16,
        num_features=4,
        device="cpu",
    )
    trained = pretrainTFM(
        model=make_model(),
        prior=prior,
        eval=[],
        criterion=nn.CrossEntropyLoss(),
        epochs=1,
        device="cpu",
    )

    with torch.no_grad():
        out = trained(torch.randn(2, 12, 4), torch.randint(0, 3, (2, 12)).float(), torch.randn(2, 5, 4))
    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()
