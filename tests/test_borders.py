import h5py
import numpy as np
import pytest
import torch
from pfns.bar_distribution import FullSupportBarDistribution, get_bucket_limits
from torch import nn

from tfmplayground.configs.models import (
    ModdedNanoTabPFNRegressorConfig,
    NanoTabICLRegressorConfig,
    NanoTabPFNRegressorConfig,
    TabFMRegressorConfig,
    TabICLRegressorConfig,
)
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFN, ModdedNanoTabPFNModel
from tfmplayground.models.nanotabicl import NanoTabICLModel
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import Prior
from tfmplayground.utils import QuantileLoss, make_bucket_borders, make_global_bucket_edges

ICL = dict(
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
NANO = dict(
    embed_dim=16,
    col_num_blocks=1,
    row_num_blocks=1,
    icl_num_blocks=1,
    col_nhead=2,
    row_nhead=2,
    icl_nhead=2,
    n_cls_cols=2,
    n_cls_rows=8,
)


def build(name, width):
    if name == "nanotabpfn":
        return NanoTabPFNModel(
            config=NanoTabPFNRegressorConfig(
                embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=width
            )
        )
    if name == "moddednanotabpfn":
        return ModdedNanoTabPFNModel(config=ModdedNanoTabPFNRegressorConfig(l=1, a=2, e=16, h=32, o=width))
    if name == "tabicl":
        return TabICLModel(config=TabICLRegressorConfig(num_quantiles=width, **ICL))
    if name == "nanotabicl":
        return NanoTabICLModel(config=NanoTabICLRegressorConfig(out_dim=width, **NANO))
    return TabFMModel(config=TabFMRegressorConfig())


SIZED = ["nanotabpfn", "moddednanotabpfn", "tabicl", "nanotabicl"]


class NormalPrior(Prior):
    def __init__(self, scale=1.0, shift=0.0, seed=0):
        self.scale, self.shift = scale, shift
        self.generator = torch.Generator().manual_seed(seed)

    def batch(self, batch_size):
        x = torch.randn(batch_size, 20, 3, generator=self.generator)
        y = torch.randn(batch_size, 20, generator=self.generator) * self.scale + self.shift
        return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]


@pytest.mark.parametrize("name", SIZED)
def test_a_sized_head_carries_one_edge_more_than_its_outputs(name):
    model = build(name, 7)
    x_train, y_train, x_test = torch.randn(2, 12, 4), torch.randn(2, 12), torch.randn(2, 5, 4)
    with torch.no_grad():
        output = model(x_train, y_train, x_test)
    assert model.borders.numel() == output.shape[-1] + 1


def test_tabfm_carries_no_edges_because_it_can_never_use_them():
    model = build("tabfm", 7)
    assert "borders" not in model.state_dict()
    x_train, y_train, x_test = torch.randn(2, 12, 4), torch.randn(2, 12), torch.randn(2, 5, 4)
    with torch.no_grad():
        assert model(x_train, y_train, x_test).shape[-1] == 1


@pytest.mark.parametrize("name", SIZED)
def test_an_unfitted_model_says_so(name):
    assert (build(name, 7).borders.diff() <= 0).any()


@pytest.mark.parametrize("name", SIZED)
def test_fitting_gives_increasing_edges(name):
    model = build(name, 7)
    model.borders = make_bucket_borders(NormalPrior(), model.borders.numel() - 1, batch_size=4, min_targets=2000)
    assert model.borders.numel() == 8
    assert (model.borders.diff() > 0).all()


@pytest.mark.parametrize("name", SIZED)
def test_the_edges_survive_a_save_and_a_load(name):
    trained = build(name, 7)
    trained.borders = make_bucket_borders(NormalPrior(), trained.borders.numel() - 1, batch_size=4, min_targets=2000)
    loaded = build(name, 7)
    loaded.load_state_dict(trained.state_dict())
    assert torch.equal(loaded.borders, trained.borders)


def test_the_vendored_modded_class_cannot_do_that():
    # its own buffer is None, which state_dict skips, so a fitted checkpoint will not load back
    vendored = ModdedNanoTabPFN(l=1, a=2, e=16, h=32, o=7)
    assert "borders" not in vendored.state_dict()
    vendored.borders = torch.linspace(-3, 3, 8)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        ModdedNanoTabPFN(l=1, a=2, e=16, h=32, o=7).load_state_dict(vendored.state_dict())


@pytest.mark.parametrize("name", SIZED)
def test_the_edges_are_fitted_on_the_values_the_loop_sees(name):
    model = build(name, 7)
    borders = make_bucket_borders(
        NormalPrior(scale=50.0, shift=100.0),
        model.borders.numel() - 1,
        batch_size=4,
        min_targets=2000,
    )
    assert borders.abs().max() < 10


@pytest.mark.parametrize("name", SIZED)
@pytest.mark.parametrize("way", ["bucket", "quantile", "scalar"])
def test_a_sized_head_trains_every_way(name, way):
    width = 1 if way == "scalar" else 9
    model = build(name, width)
    x_train, y_train, x_test, y_test = (
        torch.randn(2, 12, 4),
        torch.randn(2, 12),
        torch.randn(2, 5, 4),
        torch.randn(2, 5),
    )
    output = model(x_train, y_train, x_test)
    if way == "bucket":
        loss = FullSupportBarDistribution(get_bucket_limits(9, ys=torch.randn(5000)))(output, y_test).mean()
    elif way == "quantile":
        loss = QuantileLoss(n_quantiles=9)(output, y_test).mean()
    else:
        loss = nn.MSELoss()(output.squeeze(-1), y_test)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("way", ["bucket", "quantile"])
def test_tabfm_takes_the_scalar_way_only(way):
    model = build("tabfm", 9)
    x_train, y_train, x_test = torch.randn(2, 12, 4), torch.randn(2, 12), torch.randn(2, 5, 4)
    with torch.no_grad():
        assert model(x_train, y_train, x_test).shape[-1] != 9


def test_make_global_bucket_edges_still_reads_a_dump(tmp_path):
    path = tmp_path / "dump.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("y", data=rng.standard_normal((40, 30)).astype(np.float32))
    edges = make_global_bucket_edges(path, n_buckets=7, device=torch.device("cpu"))
    assert edges.numel() == 8
    assert (edges.diff() > 0).all()


def test_too_few_targets_are_refused():
    with pytest.raises(ValueError, match="cannot make"):
        make_bucket_borders(NormalPrior(), num_buckets=100_000, batch_size=1, min_targets=2000)


def test_repeated_targets_are_refused():
    class RepeatingPrior(Prior):
        def batch(self, batch_size):
            x = torch.randn(batch_size, 20, 3)
            pattern = torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(5)
            y = pattern.expand(batch_size, 20)
            return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]

    with pytest.raises(ValueError, match="no width"):
        make_bucket_borders(RepeatingPrior(), num_buckets=9, batch_size=4, min_targets=2000)
