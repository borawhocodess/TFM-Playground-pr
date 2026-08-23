import pytest
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground import pretrainTFM
from tfmplayground.models import (
    ModdedNanoTabPFNModel,
    NanoTabICLv2,
    NanoTabPFNModel,
    TabFMModel,
    TabICLModel,
)
from tfmplayground.priors import FunctionPrior
from tfmplayground.train import (
    check_criterion_output_kind,
    infer_criterion,
    infer_num_outputs,
    model_output_kind,
)
from tfmplayground.utils import QuantileLoss


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
    [make_nanotabpfn, make_nanotabicl, make_moddednanotabpfn, make_nanotabfm, make_tabicl],
    ids=["nanotabpfn", "nanotabicl", "moddednanotabpfn", "nanotabfm", "tabicl"],
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
    [make_nanotabpfn, make_nanotabicl, make_moddednanotabpfn, make_nanotabfm, make_tabicl],
    ids=["nanotabpfn", "nanotabicl", "moddednanotabpfn", "nanotabfm", "tabicl"],
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


def make_moddednanotabpfn_regression():
    return ModdedNanoTabPFNModel(l=2, a=2, e=16, h=32, o=NUM_BUCKETS)


def make_nanotabfm_regression():
    return TabFMModel(max_classes=0, is_classifier=False, num_outputs=1)


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
    "make_model, num_outputs, has_decoder",
    [
        (make_nanotabpfn_regression, NUM_BUCKETS, True),
        (make_nanotabicl_regression, NUM_BUCKETS, True),
        (make_moddednanotabpfn_regression, NUM_BUCKETS, True),
        (make_nanotabfm_regression, 1, False),
        (make_tabicl_regression, NUM_BUCKETS, True),
    ],
    ids=["nanotabpfn", "nanotabicl", "moddednanotabpfn", "nanotabfm", "tabicl"],
)
def test_pretrainTFM_trains_any_model_for_regression(make_model, num_outputs, has_decoder):
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

    assert hasattr(trained, "dist") is has_decoder
    with torch.no_grad():
        out = trained(torch.randn(2, 12, 4), torch.randn(2, 12), torch.randn(2, 5, 4))
    assert out.shape == (2, 5, num_outputs)
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
        (make_nanotabpfn_regression, "generic_logits"),
        (make_nanotabicl_regression, "quantiles"),
        (make_moddednanotabpfn_regression, "generic_logits"),
        (make_nanotabfm_regression, "scalar"),
        (make_tabicl_regression, "quantiles"),
    ],
    ids=["nanotabpfn", "nanotabicl", "moddednanotabpfn", "nanotabfm", "tabicl"],
)
def test_regression_criterion_follows_the_declared_head(make_model, kind):
    """
    A quantile head under a bar distribution has its outputs read as bucket logits. That trains
    against the wrong objective and never fails, so the model declares what its outputs are.
    """
    model = make_model()
    assert model_output_kind(model, "regression") == kind
    criterion = infer_criterion(model, make_regression_prior(), "cpu", problem="regression")
    criterion_type = {
        "bar_logits": FullSupportBarDistribution,
        "generic_logits": FullSupportBarDistribution,  # inferred default, a quantile loss is also allowed
        "quantiles": QuantileLoss,
        "scalar": nn.MSELoss,
    }[kind]
    assert isinstance(criterion, criterion_type)


@pytest.mark.parametrize(
    "make_model, criterion",
    [
        (
            make_nanotabicl_regression,
            FullSupportBarDistribution(torch.linspace(-3, 3, NUM_BUCKETS + 1)),
        ),
        (make_nanotabfm_regression, QuantileLoss(1)),
    ],
    ids=["quantiles-as-bar", "scalar-as-quantiles"],
)
def test_pretrainTFM_rejects_a_criterion_for_another_output_kind(make_model, criterion):
    with pytest.raises(ValueError, match="emits"):
        pretrainTFM(
            model=make_model(),
            prior=make_regression_prior(),
            criterion=criterion,
            eval=[],
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    "make_model",
    [
        make_nanotabpfn,
        make_nanotabicl,
        make_moddednanotabpfn,
        make_nanotabfm,
        make_tabicl,
        make_nanotabpfn_regression,
        make_nanotabicl_regression,
        make_moddednanotabpfn_regression,
        make_nanotabfm_regression,
        make_tabicl_regression,
    ],
)
def test_bundled_models_declare_their_output_width(make_model):
    model = make_model()
    assert model.num_outputs is not None
    assert infer_num_outputs(model) == model.num_outputs


@pytest.mark.parametrize(
    "make_model",
    [make_nanotabpfn, make_nanotabicl, make_moddednanotabpfn, make_nanotabfm, make_tabicl],
)
def test_classification_models_declare_class_logits(make_model):
    assert model_output_kind(make_model(), "classification") == "class_logits"


def test_scalar_regression_keeps_its_contract_when_data_parallel_wraps_the_model():
    trained = pretrainTFM(
        model=make_nanotabfm_regression(),
        prior=make_regression_prior(),
        eval=[],
        epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        device="cpu",
        multi_gpu=True,
    )

    assert infer_num_outputs(trained) == 1


def test_tabicl_evaluation_disables_numeric_attention_dropout():
    torch.manual_seed(0)
    model = make_tabicl(dropout=0.2)
    model.eval()
    X_train = torch.randn(2, 12, 4)
    y_train = torch.randint(0, 3, (2, 12)).float()
    X_test = torch.randn(2, 5, 4)

    with torch.no_grad():
        first = model(X_train, y_train, X_test)
        second = model(X_train, y_train, X_test)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "criterion",
    [
        lambda: FullSupportBarDistribution(torch.linspace(-3, 3, NUM_BUCKETS + 1)),
        lambda: QuantileLoss(NUM_BUCKETS),
    ],
    ids=["bar", "quantiles"],
)
def test_a_generic_head_takes_either_regression_criterion(criterion):
    """
    The tabpfn lineage head is n logits whose meaning the criterion decides, so a caller who picks
    a quantile loss deliberately is not making a mistake. Only the structural kinds are exact.
    """
    model = make_nanotabpfn_regression()
    assert model.output_kinds["regression"] == "generic_logits"

    check_criterion_output_kind(model, criterion(), "generic_logits")


def test_a_generic_head_still_refuses_a_structural_criterion():
    """Generic does not mean anything goes: a scalar loss reads the channels differently."""
    model = make_nanotabpfn_regression()

    with pytest.raises(ValueError, match="emits generic_logits"):
        check_criterion_output_kind(model, nn.MSELoss(), "generic_logits")
