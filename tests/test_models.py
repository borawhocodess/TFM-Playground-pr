import pytest
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground import pretrainTFM
from tfmplayground.models import (
    ModdedNanoTabPFNModel,
    NanoTabICLv2,
    NanoTabPFNModel,
    TabDPTModel,
    TabFMModel,
    TabICLModel,
)
from tfmplayground.priors import FunctionPrior
from tfmplayground.train import check_criterion_kind, infer_criterion, infer_num_outputs
from tfmplayground.utils import FixedBinLoss, QuantileLoss


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


def make_tabicl(max_classes=3, **kwargs):
    """The official model, kept as small as its architecture allows."""
    return TabICLModel(
        max_classes=max_classes,
        embed_dim=32,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        col_num_inds=8,
        row_num_cls=2,
        **kwargs,
    )


@pytest.mark.parametrize(
    "make_model",
    [make_nanotabpfn, make_nanotabicl, make_nanotabdpt, make_moddednanotabpfn, make_nanotabfm, make_tabicl],
    ids=["nanotabpfn", "nanotabicl", "nanotabdpt", "moddednanotabpfn", "nanotabfm", "tabicl"],
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
    sep = num_datapoints // 2
    return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


@pytest.mark.parametrize(
    "make_model",
    [make_nanotabpfn, make_nanotabicl, make_nanotabdpt, make_moddednanotabpfn, make_nanotabfm, make_tabicl],
    ids=["nanotabpfn", "nanotabicl", "nanotabdpt", "moddednanotabpfn", "nanotabfm", "tabicl"],
)
def test_pretrainTFM_swaps_any_model(make_model):
    """Every model drops into pretrainTFM as-is and comes back trained."""
    torch.manual_seed(0)
    model = make_model()
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    prior = FunctionPrior(
        get_batch_function=get_three_class_batch,
        num_datapoints_max=16,
        num_features=4,
        device="cpu",
    )
    trained = pretrainTFM(
        model=model,
        prior=prior,
        eval=[],
        criterion=nn.CrossEntropyLoss(),
        epochs=1,
        steps_per_epoch=2,
        batch_size=2,
        device="cpu",
    )

    with torch.no_grad():
        out = trained(torch.randn(2, 12, 4), torch.randint(0, 3, (2, 12)).float(), torch.randn(2, 5, 4))
    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, trained.parameters(), strict=True)
    )


NUM_BUCKETS = 8


def make_nanotabpfn_regression():
    return NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=2,
        mlp_hidden_size=32,
        num_layers=2,
        num_outputs=NUM_BUCKETS,
    )


def make_nanotabicl_regression():
    return NanoTabICLv2(
        max_classes=0,  # 0 swaps the class lookup for a linear encoder over the scalar target
        out_dim=NUM_BUCKETS,
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


def make_nanotabdpt_regression():
    return TabDPTModel(
        dropout=0.0,
        n_out=NUM_BUCKETS,
        regression_bin_count=NUM_BUCKETS,
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
        classification=False,
    )


def make_moddednanotabpfn_regression():
    return ModdedNanoTabPFNModel(l=2, a=2, e=16, h=32, o=NUM_BUCKETS)


def make_nanotabfm_regression():
    return TabFMModel(max_classes=0, is_classifier=False, num_outputs=NUM_BUCKETS)


def make_tabicl_regression():
    return make_tabicl(max_classes=0, num_quantiles=NUM_BUCKETS)


def get_continuous_batch(batch_size, num_datapoints, num_features, problem=None):
    x = torch.randn(batch_size, num_datapoints, num_features)
    y = 2.0 * x[:, :, 0] - x[:, :, 1]
    sep = num_datapoints // 2
    return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


def make_regression_prior():
    return FunctionPrior(
        get_batch_function=get_continuous_batch,
        num_datapoints_max=16,
        num_features=4,
        device="cpu",
        problem="regression",
    )


@pytest.mark.parametrize(
    "make_model",
    [
        make_nanotabpfn_regression,
        make_nanotabicl_regression,
        make_nanotabdpt_regression,
        make_moddednanotabpfn_regression,
        make_nanotabfm_regression,
        make_tabicl_regression,
    ],
    ids=["nanotabpfn", "nanotabicl", "nanotabdpt", "moddednanotabpfn", "nanotabfm", "tabicl"],
)
def test_pretrainTFM_trains_any_model_for_regression(make_model):
    """Regression goes through every model too, not just the one the other regression tests use."""
    torch.manual_seed(0)
    model = make_model()
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    trained = pretrainTFM(
        model=model,
        prior=make_regression_prior(),
        eval=[],
        epochs=1,
        steps_per_epoch=2,
        batch_size=2,
        device="cpu",
    )

    assert trained.dist is not None
    with torch.no_grad():
        out = trained(torch.randn(2, 12, 4), torch.randn(2, 12), torch.randn(2, 5, 4))
    assert out.shape == (2, 5, NUM_BUCKETS)
    assert torch.isfinite(out).all()
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, trained.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    "make_model, problem",
    [
        (make_nanotabicl, "regression"),
        (make_nanotabfm, "regression"),
        (make_tabicl, "regression"),
        (make_nanotabicl_regression, "classification"),
        (make_tabicl_regression, "classification"),
    ],
    ids=["nanotabicl", "nanotabfm", "tabicl", "nanotabicl-reversed", "tabicl-reversed"],
)
def test_pretrainTFM_rejects_a_model_built_for_the_other_problem(make_model, problem):
    """
    A model built for classification looks y up in a class lookup, so continuous targets truncate
    to indices. cpu and cuda raise on the out of range ones, mps silently returns zeros and trains
    on a garbage y embedding, so refuse the pairing before any of that. The reverse direction is
    refused too, a regression head has no business under a classification loss.
    """
    prior = make_regression_prior() if problem == "regression" else None
    with pytest.raises(ValueError, match="was built for"):
        pretrainTFM(
            model=make_model(),
            prior=prior,
            problem=problem,
            eval=[],
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    "make_model, kind",
    [
        (make_nanotabpfn_regression, "bar"),
        (make_nanotabicl_regression, "quantiles"),
        (make_nanotabdpt_regression, "fixed_bin_logits"),
        (make_moddednanotabpfn_regression, "bar"),
        (make_nanotabfm_regression, "bar"),
        (make_tabicl_regression, "quantiles"),
    ],
    ids=["nanotabpfn", "nanotabicl", "nanotabdpt", "moddednanotabpfn", "nanotabfm", "tabicl"],
)
def test_regression_criterion_follows_the_declared_head(make_model, kind):
    """
    A quantile head under a bar distribution has its outputs read as bucket logits. That trains
    against the wrong objective and never fails, so the model declares what its outputs are.
    """
    model = make_model()
    assert model.output_kind == kind
    criterion = infer_criterion(model, make_regression_prior(), "cpu", problem="regression")
    expected = {"quantiles": QuantileLoss, "fixed_bin_logits": FixedBinLoss, "bar": FullSupportBarDistribution}
    assert isinstance(criterion, expected[kind])
    if kind == "fixed_bin_logits":
        # the model's own bins, not edges fitted from the prior, which would mean something else
        assert torch.equal(criterion.borders.cpu(), model.regression_borders())


def test_fixed_bins_are_not_refitted_from_the_prior():
    """
    tabdpt's regression channels mean evenly spaced bins between the bounds it was built with.
    Fitting edges from the prior instead hands those channels ranges the model never meant, and
    nothing about the shapes would object.
    """
    model = TabDPTModel(
        dropout=0.0,
        n_out=NUM_BUCKETS,
        regression_bin_count=NUM_BUCKETS,
        regression_bin_min=-7.0,  # deliberately nothing like what a z normalised prior would fit
        regression_bin_max=7.0,
        nhead=2,
        nhid=32,
        ninp=16,
        nlayers=2,
        num_features=8,
        base_len=32,
        max_len=64,
        y_encoder_dim=8,
        classification=False,
    )
    criterion = infer_criterion(model, make_regression_prior(), "cpu", problem="regression")

    assert criterion.borders[0].item() == pytest.approx(-7.0)
    assert criterion.borders[-1].item() == pytest.approx(7.0)
    assert criterion.borders.numel() - 1 == NUM_BUCKETS


def test_a_model_without_its_borders_cannot_claim_fixed_bins():
    """The declaration is a promise to say what the bins are, so an empty promise is an error."""
    model = make_nanotabpfn_regression()
    model.output_kind = "fixed_bin_logits"

    with pytest.raises(ValueError, match="no regression_borders"):
        infer_criterion(model, make_regression_prior(), "cpu", problem="regression")


def test_tabicl_predicts_deterministically_with_dropout_on():
    """
    The wrapper forces the upstream training path, which means forcing training mode on the whole
    tree, which is where dropout lives. If any of it stays on, the same input gives two answers.
    """
    torch.manual_seed(0)
    model = make_tabicl(max_classes=3, dropout=0.5).eval()
    X_train, y_train = torch.randn(2, 12, 5), torch.randint(0, 3, (2, 12)).float()
    X_test = torch.randn(2, 5, 5)

    with torch.no_grad():
        first, second = model(X_train, y_train, X_test), model(X_train, y_train, X_test)

    assert torch.equal(first, second)
    assert all(not module.training for module in model.modules())  # modes restored


@pytest.mark.parametrize(
    "make_model, criterion, message",
    [
        (
            make_nanotabicl_regression,
            lambda: FullSupportBarDistribution(torch.linspace(-3, 3, 9)),
            "needs QuantileLoss",
        ),
        (make_nanotabdpt_regression, lambda: QuantileLoss(NUM_BUCKETS), "needs FixedBinLoss"),
        (make_nanotabicl, lambda: QuantileLoss(3), "built for classification"),
    ],
    ids=["bar-into-quantiles", "quantiles-into-fixed-bins", "regression-loss-into-class-head"],
)
def test_an_explicit_criterion_must_match_the_head(make_model, criterion, message):
    """
    Inference reads output_kind, but a criterion handed in by the caller used to go unchecked, so
    the very bug output_kind exists to stop was still reachable through the front door.
    """
    with pytest.raises(ValueError, match=message):
        pretrainTFM(
            model=make_model(),
            prior=make_regression_prior(),
            eval=[],
            criterion=criterion(),
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
            device="cpu",
        )


def test_a_generic_head_accepts_either_regression_criterion():
    """The tabpfn lineage head is n logits whose meaning is the criterion's, so do not over police it."""
    model = make_nanotabpfn_regression()
    assert model.output_kind == "bar"

    for criterion in (FullSupportBarDistribution(torch.linspace(-3, 3, NUM_BUCKETS + 1)), QuantileLoss(NUM_BUCKETS)):
        check_criterion_kind(model, criterion, "regression")


def test_scalar_heads_are_refused_rather_than_mistrained():
    """
    A scalar head emits (batch, rows, 1) against targets of (batch, rows), which mse broadcasts
    into (batch, rows, rows) without complaint. Telling someone to pass a criterion invited that.
    """
    model = TabFMModel(max_classes=0, is_classifier=False)  # upstream's shape, one value per row
    assert model.output_kind == "scalar"

    with pytest.raises(NotImplementedError, match="does not make it safe"):
        check_criterion_kind(model, torch.nn.MSELoss(), "regression")
    with pytest.raises(NotImplementedError, match="scalar regression head"):
        infer_criterion(model, make_regression_prior(), "cpu", problem="regression")


def test_declared_output_counts_are_used_without_probing():
    """Probing runs a forward pass, which a custom model carrying running statistics would notice."""
    model = make_nanotabicl_regression()
    assert model.num_outputs == NUM_BUCKETS

    model.forward = None  # probing would explode, the declaration must be enough
    assert infer_num_outputs(model) == NUM_BUCKETS
