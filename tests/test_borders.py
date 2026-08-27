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
from tfmplayground.interface import TabularRegressor
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFN, ModdedNanoTabPFNModel
from tfmplayground.models.nanotabicl import NanoTabICLModel
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import Prior
from tfmplayground.utils import QuantileLoss, ScalarMSELoss, make_bucket_borders, make_global_bucket_edges

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
    model.borders = make_bucket_borders(
        NormalPrior(), model.borders.numel() - 1, batch_size=4, min_targets=2000, outlier_threshold=6.0
    )
    assert model.borders.numel() == 8
    assert (model.borders.diff() > 0).all()


@pytest.mark.parametrize("name", SIZED)
def test_the_edges_survive_a_save_and_a_load(name):
    trained = build(name, 7)
    trained.borders = make_bucket_borders(
        NormalPrior(), trained.borders.numel() - 1, batch_size=4, min_targets=2000, outlier_threshold=6.0
    )
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
        outlier_threshold=6.0,
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
        loss = ScalarMSELoss()(output, y_test).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_tabfm_trains_the_scalar_way():
    model = build("tabfm", 1)
    x_train, y_train, x_test, y_test = (
        torch.randn(2, 12, 4),
        torch.randn(2, 12),
        torch.randn(2, 5, 4),
        torch.randn(2, 5),
    )
    output = model(x_train, y_train, x_test)
    loss = ScalarMSELoss()(output, y_test).mean()
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
        make_bucket_borders(NormalPrior(), num_buckets=100_000, batch_size=1, min_targets=2000, outlier_threshold=6.0)


def test_repeated_targets_are_refused():
    class RepeatingPrior(Prior):
        def batch(self, batch_size):
            x = torch.randn(batch_size, 20, 3)
            pattern = torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(5)
            y = pattern.expand(batch_size, 20)
            return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]

    with pytest.raises(ValueError, match="no width"):
        make_bucket_borders(RepeatingPrior(), num_buckets=9, batch_size=4, min_targets=2000, outlier_threshold=6.0)


def test_exactly_one_target_for_each_bucket_is_allowed():
    class OneEachPrior(Prior):
        def batch(self, batch_size):
            x = torch.randn(batch_size, 10, 3)
            y = torch.randn(batch_size, 10)
            return x[:, :5], y[:, :5], x[:, 5:], y[:, 5:]

    borders = make_bucket_borders(OneEachPrior(), num_buckets=10, batch_size=1, min_targets=10, outlier_threshold=6.0)
    assert borders.numel() == 11
    assert (borders.diff() > 0).all()


def test_a_nonfinite_target_drops_only_its_own_table():
    class OneBadTablePrior(Prior):
        def batch(self, batch_size):
            x = torch.randn(batch_size, 20, 3)
            y = torch.randn(batch_size, 20)
            y[0, 0] = float("nan")
            y[0, 1] = float("inf")
            return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]

    borders = make_bucket_borders(
        OneBadTablePrior(), num_buckets=9, batch_size=4, min_targets=400, outlier_threshold=6.0
    )
    assert torch.isfinite(borders).all()
    assert (borders.diff() > 0).all()


def test_a_batch_of_one_bad_table_leaves_nothing():
    # the mean and the standard deviation are taken for each table, so one bad value spoils that table
    class AllBadPrior(Prior):
        def batch(self, batch_size):
            x = torch.randn(batch_size, 20, 3)
            y = torch.randn(batch_size, 20)
            y[:, 0] = float("nan")
            return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]

    with pytest.raises(ValueError, match="0 targets cannot make"):
        make_bucket_borders(AllBadPrior(), num_buckets=9, batch_size=1, min_targets=200, outlier_threshold=6.0)


def test_a_rare_extreme_table_does_not_widen_the_borders():
    class RareExtremePrior(Prior):
        def __init__(self):
            self.draws = 0

        def batch(self, batch_size):
            self.draws += 1
            x = torch.randn(batch_size, 40, 3)
            y = torch.randn(batch_size, 40)
            if self.draws % 50 == 0:
                y = y * 500.0
            return x[:, :24], y[:, :24], x[:, 24:], y[:, 24:]

    borders = make_bucket_borders(
        RareExtremePrior(), num_buckets=20, batch_size=1, min_targets=8000, outlier_threshold=6.0
    )
    assert borders.abs().max() < 10


def test_a_wider_threshold_gives_wider_borders():
    class NoisyPrior(Prior):
        def __init__(self, seed):
            self.generator = torch.Generator().manual_seed(seed)

        def batch(self, batch_size):
            x = torch.randn(batch_size, 40, 3, generator=self.generator)
            y = torch.randn(batch_size, 40, generator=self.generator)
            return x[:, :24], y[:, :24], x[:, 24:], y[:, 24:]

    narrow = make_bucket_borders(NoisyPrior(0), 20, 1, 8000, outlier_threshold=3.0)
    wide = make_bucket_borders(NoisyPrior(0), 20, 1, 8000, outlier_threshold=6.0)
    assert wide.abs().max() > narrow.abs().max()


def test_the_scalar_loss_matches_the_library_one():
    logits = torch.randn(2, 12, 1)
    target = torch.randn(2, 12)
    ours = ScalarMSELoss()(logits, target).mean()
    theirs = nn.MSELoss()(logits.squeeze(-1), target)
    assert torch.allclose(ours, theirs)


def test_the_scalar_loss_keeps_one_number_for_each_row():
    assert ScalarMSELoss()(torch.randn(2, 12, 1), torch.randn(2, 12)).shape == (2, 12)


def test_the_scalar_loss_refuses_a_head_that_is_not_scalar():
    with pytest.raises(RuntimeError):
        ScalarMSELoss()(torch.randn(2, 12, 9), torch.randn(2, 12))


@pytest.mark.parametrize("name", SIZED)
def test_a_scalar_head_predicts_through_the_same_decoder(name):
    model = build(name, 1)
    model.config.head = "scalar"
    regressor = TabularRegressor(model, device="cpu")
    rng = np.random.default_rng(0)
    regressor.fit(rng.standard_normal((20, 4)), rng.standard_normal(20))
    predictions = regressor.predict(rng.standard_normal((10, 4)))
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()


def test_tabfm_predicts_the_scalar_way():
    regressor = TabularRegressor(build("tabfm", 1), device="cpu")
    rng = np.random.default_rng(0)
    regressor.fit(rng.standard_normal((20, 4)), rng.standard_normal(20))
    assert regressor.predict(rng.standard_normal((10, 4))).shape == (10,)
