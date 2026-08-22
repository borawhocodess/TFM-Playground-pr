import h5py
import numpy as np
import pytest
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground import TabularClassifier, TabularRegressor, pretrainTFM
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, OpenMLEvaluationCallback
from tfmplayground.models import NanoTabPFNModel
from tfmplayground.prior import MAX_NUM_CLASSES, DictPrior, DumpPrior, FunctionPrior, SCMPrior
from tfmplayground.train import default_prior, infer_criterion, infer_num_outputs
from tfmplayground.utils import QuantileLoss, dump_targets, fetch_dump, make_global_bucket_edges


def make_tiny_model(num_outputs):
    return NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=2,
        mlp_hidden_size=32,
        num_layers=2,
        num_outputs=num_outputs,
    )


def get_classification_batch(batch_size, num_datapoints, num_features):
    x = torch.randn(batch_size, num_datapoints, num_features)
    y = (x.sum(dim=-1) > 0).float()
    sep = num_datapoints // 2
    return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


def make_regression_dump(path, num_tables=8, num_datapoints=16, num_features=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((num_tables, num_datapoints, num_features)).astype("f4")
    y = X.sum(axis=-1).astype("f4")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.create_dataset("num_features", data=np.full(num_tables, num_features, dtype="i4"))
        f.create_dataset("num_datapoints", data=np.full(num_tables, num_datapoints, dtype="i4"))
        f.create_dataset("train_test_split_index", data=np.full(num_tables, num_datapoints // 2, dtype="i4"))
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())


def make_classification_dump(path, num_tables=8, num_datapoints=16, num_features=3, max_num_classes=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((num_tables, num_datapoints, num_features)).astype("f4")
    y = rng.integers(0, max_num_classes, size=(num_tables, num_datapoints)).astype("f4")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.create_dataset("num_features", data=np.full(num_tables, num_features, dtype="i4"))
        f.create_dataset("num_datapoints", data=np.full(num_tables, num_datapoints, dtype="i4"))
        f.create_dataset("train_test_split_index", data=np.full(num_tables, num_datapoints // 2, dtype="i4"))
        f.create_dataset("max_num_classes", data=np.array((max_num_classes,)))
        f.create_dataset("problem_type", data="classification", dtype=h5py.string_dtype())


def test_pretrainTFM_classification_returns_usable_model():
    """One call with a model and a prior gives back a model that plugs into the classifier interface."""
    torch.manual_seed(0)
    prior = FunctionPrior(
        get_batch_function=get_classification_batch,
        num_datapoints_max=16,
        num_features=3,
        device="cpu",
    )
    trained = pretrainTFM(
        model=make_tiny_model(num_outputs=2),
        prior=prior,
        eval=[],
        criterion=nn.CrossEntropyLoss(),
        epochs=2,
        steps_per_epoch=2,
        batch_size=4,
        device="cpu",
    )

    classifier = TabularClassifier(trained, device="cpu")
    rng = np.random.default_rng(0)
    classifier.fit(rng.standard_normal((20, 3)), rng.integers(0, 2, size=20))
    probabilities = classifier.predict_proba(rng.standard_normal((5, 3)))
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_pretrainTFM_default_model_head_is_fixed(tmp_path):
    """Only a prior given, the default model keeps its 10 class head no matter what the dump says."""
    torch.manual_seed(0)
    dump = tmp_path / "tiny_classification.h5"
    make_classification_dump(dump)
    prior = DumpPrior(filename=str(dump))

    trained = pretrainTFM(prior=prior, eval=[], epochs=1, steps_per_epoch=2, batch_size=4, device="cpu")

    assert infer_num_outputs(trained) == 10


def test_problem_flag_forces_classification():
    """An in-memory prior carries no problem_type, the flag picks cross entropy where inference guesses quantiles."""
    torch.manual_seed(0)
    prior = FunctionPrior(
        get_batch_function=get_classification_batch,
        num_datapoints_max=16,
        num_features=3,
        device="cpu",
    )
    model = make_tiny_model(num_outputs=2)

    assert isinstance(infer_criterion(model, prior, "cpu"), QuantileLoss)
    assert isinstance(infer_criterion(model, prior, "cpu", problem="classification"), nn.CrossEntropyLoss)

    trained = pretrainTFM(
        model=model,
        prior=prior,
        eval=[],
        problem="classification",
        epochs=1,
        steps_per_epoch=2,
        batch_size=4,
        device="cpu",
    )
    with torch.no_grad():
        out = trained(torch.randn(1, 8, 3), torch.randint(0, 2, (1, 8)).float(), torch.randn(1, 4, 3))
    assert torch.isfinite(out).all()


def test_problem_flag_contradicting_prior_raises(tmp_path):
    dump = tmp_path / "tiny_classification.h5"
    make_classification_dump(dump)
    prior = DumpPrior(filename=str(dump), device="cpu")

    with pytest.raises(ValueError, match="the prior says"):
        pretrainTFM(prior=prior, problem="regression", device="cpu")


def test_problem_flag_rejects_unknown_values():
    with pytest.raises(ValueError, match="must be one of"):
        pretrainTFM(problem="clustering", device="cpu")


def test_criterion_contradicting_problem_raises(tmp_path):
    """A criterion declares a side, starting a run where it disagrees with the prior or flag refuses loudly."""
    make_classification_dump(tmp_path / "cls.h5")
    make_regression_dump(tmp_path / "reg.h5")
    cls_prior = DumpPrior(filename=str(tmp_path / "cls.h5"), device="cpu")
    reg_prior = DumpPrior(filename=str(tmp_path / "reg.h5"), device="cpu")
    model = make_tiny_model(num_outputs=3)

    with pytest.raises(ValueError, match="is for classification"):
        pretrainTFM(model=model, prior=reg_prior, criterion=nn.CrossEntropyLoss(), device="cpu")
    with pytest.raises(ValueError, match="is for regression"):
        pretrainTFM(model=model, prior=cls_prior, criterion=QuantileLoss(3), device="cpu")
    with pytest.raises(ValueError, match="is for regression"):
        pretrainTFM(
            model=model, prior=cls_prior, criterion=FullSupportBarDistribution(torch.linspace(-3, 3, 4)), device="cpu"
        )

    in_memory = FunctionPrior(
        get_batch_function=get_classification_batch,
        num_datapoints_max=16,
        num_features=3,
        device="cpu",
    )
    with pytest.raises(ValueError, match="is for regression"):
        pretrainTFM(model=model, prior=in_memory, problem="classification", criterion=QuantileLoss(3), device="cpu")


def test_eval_direction_contradicting_problem_raises(tmp_path):
    """A directional eval callback pointed the wrong way fails before training instead of after an epoch."""
    make_classification_dump(tmp_path / "cls.h5")
    prior = DumpPrior(filename=str(tmp_path / "cls.h5"), device="cpu")

    with pytest.raises(ValueError, match="set up for regression"):
        pretrainTFM(
            prior=prior,
            eval=OpenMLEvaluationCallback(TOY_TASKS_REGRESSION, classification=False, device="cpu"),
            device="cpu",
        )


def test_model_head_contradicting_criterion_raises(tmp_path):
    """An explicit regression criterion must have exactly as many buckets or quantiles as the model has outputs."""
    make_regression_dump(tmp_path / "reg.h5")
    prior = DumpPrior(filename=str(tmp_path / "reg.h5"), device="cpu")

    with pytest.raises(ValueError, match="100 buckets but the model has 8"):
        pretrainTFM(
            model=make_tiny_model(num_outputs=8),
            prior=prior,
            criterion=FullSupportBarDistribution(torch.linspace(-3, 3, 101)),
            device="cpu",
        )
    with pytest.raises(ValueError, match="99 quantiles but the model has 8"):
        pretrainTFM(model=make_tiny_model(num_outputs=8), prior=prior, criterion=QuantileLoss(99), device="cpu")


def test_model_head_contradicting_prior_classes_raises(tmp_path):
    """A prior with more classes than the model has outputs would crash mid-training, refuse upfront instead."""
    make_classification_dump(tmp_path / "cls15.h5", max_num_classes=15)
    prior = DumpPrior(filename=str(tmp_path / "cls15.h5"), device="cpu")

    with pytest.raises(ValueError, match="15 classes but the model has 3"):
        pretrainTFM(model=make_tiny_model(num_outputs=3), prior=prior, device="cpu")


def test_fetch_dump_prefers_cache(tmp_path):
    """A dump already sitting in the cache is returned without touching the network."""
    (tmp_path / "dump.h5").write_bytes(b"cached")
    path = fetch_dump("http://example.invalid/dump.h5", cache_dir=tmp_path)
    assert path.read_bytes() == b"cached"


def test_pretrainTFM_regression_dump_infers_criterion(tmp_path):
    """A regression dump prior needs no explicit criterion, a bar distribution is fitted on the dump."""
    torch.manual_seed(0)
    dump = tmp_path / "tiny_regression.h5"
    make_regression_dump(dump)
    prior = DumpPrior(filename=str(dump), device="cpu")

    model = make_tiny_model(num_outputs=8)
    criterion = infer_criterion(model, prior, device="cpu")
    assert criterion.borders.shape == (9,)
    assert torch.equal(criterion.borders, make_global_bucket_edges(prior, n_buckets=8, device="cpu"))

    trained = pretrainTFM(model=model, prior=prior, eval=[], epochs=2, steps_per_epoch=2, batch_size=4, device="cpu")
    assert trained.dist.borders.shape == (9,)

    regressor = TabularRegressor(trained, device="cpu")
    rng = np.random.default_rng(0)
    regressor.fit(rng.standard_normal((20, 3)), rng.standard_normal(20))
    predictions = regressor.predict(rng.standard_normal((5, 3)))
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()


def test_function_prior_forwards_the_problem():
    """A get_batch function that takes a problem gets told which side it is sampling for."""
    seen = []

    def spy(batch_size, num_datapoints, num_features, problem="classification"):
        seen.append(problem)
        return get_classification_batch(batch_size, num_datapoints, num_features)

    prior = FunctionPrior(
        get_batch_function=spy,
        num_datapoints_max=16,
        num_features=3,
        device="cpu",
        problem="regression",
    )

    assert prior.problem_type == "regression"
    prior.batch(2)
    prior.batch(2)
    assert seen == ["regression", "regression"]


def test_function_prior_leaves_the_problem_out_when_unset():
    """Batch functions that know nothing about problems keep working untouched."""
    prior = FunctionPrior(
        get_batch_function=get_classification_batch,
        num_datapoints_max=16,
        num_features=3,
        device="cpu",
    )

    assert prior.problem_type is None
    x_train, y_train, x_test, y_test = prior.batch(2)
    assert x_train.shape == (2, 8, 3)
    assert x_test.shape == (2, 8, 3)


def make_scm_regression_prior():
    return SCMPrior(num_datapoints_max=160, num_features=4, problem="regression", device="cpu")


def test_bucket_edges_fit_on_a_prior_with_no_dump_behind_it():
    """Bucket edges come off sampled batches when there is no file to read them from."""
    edges = make_global_bucket_edges(
        make_scm_regression_prior(), n_buckets=8, device="cpu", batch_size=2, max_batches=2
    )

    assert edges.shape == (9,)
    assert bool(torch.all(edges[1:] > edges[:-1]))


def test_regression_prior_without_a_dump_infers_a_bar_distribution():
    """A prior that declares regression earns a fitted bar distribution, not the quantile fallback."""
    criterion = infer_criterion(make_tiny_model(num_outputs=8), make_scm_regression_prior(), "cpu")

    assert isinstance(criterion, FullSupportBarDistribution)
    assert criterion.borders.shape == (9,)


def test_pretrainTFM_trains_on_a_sampled_regression_prior():
    """The whole path end to end: sampled batches in, a model carrying its fitted distribution out."""
    torch.manual_seed(0)

    trained = pretrainTFM(
        model=make_tiny_model(num_outputs=8),
        prior=make_scm_regression_prior(),
        eval=[],
        epochs=1,
        steps_per_epoch=2,
        batch_size=2,
        device="cpu",
    )

    assert isinstance(trained.dist, FullSupportBarDistribution)

    regressor = TabularRegressor(trained, device="cpu")
    rng = np.random.default_rng(0)
    regressor.fit(rng.standard_normal((20, 4)), rng.standard_normal(20))
    assert np.isfinite(regressor.predict(rng.standard_normal((5, 4)))).all()


@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_default_prior_samples_the_scm_on_the_fly(problem):
    """The default prior needs no dump and hands the problem down to our own get_batch."""
    prior = default_prior("cpu", problem)

    assert isinstance(prior, SCMPrior)
    assert getattr(prior, "filename", None) is None
    assert prior.problem_type == problem

    x_train, y_train, x_test, y_test = prior.batch(2)
    assert x_train.shape[0] == 2
    assert x_train.shape[1] > 0
    assert x_test.shape[1] > 0
    assert y_train.shape[1] == x_train.shape[1]
    assert y_test.shape[1] == x_test.shape[1]


def test_default_classification_prior_fits_the_default_head():
    """The scm caps itself at ten classes, which is exactly what the default model head holds."""
    prior = default_prior("cpu", "classification")
    assert prior.max_num_classes == MAX_NUM_CLASSES

    _, y_train, _, y_test = prior.batch(2)
    labels = torch.cat([y_train, y_test], dim=1).unique()
    assert labels.numel() <= MAX_NUM_CLASSES
    assert torch.equal(labels, torch.arange(labels.numel(), dtype=labels.dtype))


def test_dict_prior_restarts_a_finite_loader():
    class FiniteLoader:
        def __iter__(self):
            for marker in (1.0, 2.0):
                x = torch.full((2, 4, 1), marker)
                y = torch.zeros(2, 4)
                yield {"x": x, "y": y, "target_y": y, "train_test_split_index": 2}

    prior = DictPrior(FiniteLoader(), problem="classification", max_num_classes=2)
    markers = [prior.batch(2)[0][0, 0, 0].item() for _ in range(3)]

    assert markers == [1.0, 2.0, 1.0]


def test_dump_targets_skips_padding_rows(tmp_path):
    dump = tmp_path / "padded_regression.h5"
    first = np.array([0.0, 2.0, 100.0, 200.0])
    second = np.array([10.0, 12.0, 14.0, 20.0, 22.0])
    with h5py.File(dump, "w") as f:
        f.create_dataset(
            "y",
            data=np.array(
                [
                    [*first, 10_000.0, 10_000.0],
                    [*second, 10_000.0],
                ],
                dtype="f4",
            ),
        )
        f.create_dataset("num_datapoints", data=np.array([first.size, second.size], dtype="i4"))

    targets = dump_targets(dump, max_y=9)

    expected = np.concatenate([(t - t.mean()) / (t.std(ddof=1) + 1e-8) for t in (first, second)])
    assert targets.size == first.size + second.size
    np.testing.assert_allclose(targets, expected, rtol=1e-5)


def test_quantile_trained_model_carries_its_decoder():
    prior = SCMPrior(num_datapoints_max=160, num_features=4, problem="regression", device="cpu")
    trained = pretrainTFM(
        model=make_tiny_model(5),
        prior=prior,
        criterion=QuantileLoss(5),
        eval=[],
        epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        device="cpu",
    )

    assert isinstance(trained.dist, QuantileLoss)
    regressor = TabularRegressor(trained, device="cpu")
    regressor.fit(np.random.default_rng(0).standard_normal((6, 4)), np.arange(6, dtype=float))
    assert regressor.predict(np.ones((3, 4))).shape == (3,)


def test_single_row_context_does_not_poison_the_model():
    class OneRowPrior(SCMPrior):
        def batch(self, batch_size):
            x_train, y_train, x_test, y_test = super().batch(batch_size)
            return x_train[:, :1], y_train[:, :1], x_test, y_test

    prior = OneRowPrior(num_datapoints_max=160, num_features=4, problem="regression", device="cpu")
    x_train, y_train, _, _ = prior.batch(2)
    assert torch.isfinite(y_train).all()
    assert not torch.isfinite(y_train.std(dim=1, keepdim=True)).any()

    with pytest.raises(RuntimeError, match="no finite batches"):
        pretrainTFM(
            model=make_tiny_model(4),
            prior=prior,
            criterion=FullSupportBarDistribution(torch.linspace(-3, 3, 5)),
            eval=[],
            epochs=1,
            steps_per_epoch=2,
            batch_size=2,
            device="cpu",
        )
