import h5py
import numpy as np
import pytest
import torch
from torch import nn

from tfmplayground import TabularClassifier, TabularRegressor, pretrainTFM
from tfmplayground.external_priors import PriorDataLoader, PriorDumpDataLoader
from tfmplayground.models import NanoTabPFNModel
from tfmplayground.train import infer_criterion, infer_num_outputs
from tfmplayground.utils import QuantileLoss, fetch_dump


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
    return dict(x=x, y=y, target_y=y, train_test_split_index=num_datapoints // 2)


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
    prior = PriorDataLoader(
        get_batch_function=get_classification_batch,
        num_steps=2,
        batch_size=4,
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
        device="cpu",
    )

    classifier = TabularClassifier(trained, device="cpu")
    rng = np.random.default_rng(0)
    classifier.fit(rng.standard_normal((20, 3)), rng.integers(0, 2, size=20))
    probabilities = classifier.predict_proba(rng.standard_normal((5, 3)))
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_pretrainTFM_defaults_model_from_prior(tmp_path):
    """Only a prior given, the model defaults to a nanotabpfn sized off the prior's classes."""
    torch.manual_seed(0)
    dump = tmp_path / "tiny_classification.h5"
    make_classification_dump(dump)
    prior = PriorDumpDataLoader(filename=str(dump), num_steps=2, batch_size=4)

    trained = pretrainTFM(prior=prior, eval=[], epochs=1, device="cpu")

    assert infer_num_outputs(trained) == 3


def test_problem_flag_forces_classification():
    """An in-memory prior carries no problem_type, the flag picks cross entropy where inference guesses quantiles."""
    torch.manual_seed(0)
    prior = PriorDataLoader(
        get_batch_function=get_classification_batch,
        num_steps=2,
        batch_size=4,
        num_datapoints_max=16,
        num_features=3,
        device="cpu",
    )
    model = make_tiny_model(num_outputs=2)

    assert isinstance(infer_criterion(model, prior, "cpu"), QuantileLoss)
    assert isinstance(infer_criterion(model, prior, "cpu", problem="classification"), nn.CrossEntropyLoss)

    trained = pretrainTFM(model=model, prior=prior, eval=[], problem="classification", epochs=1, device="cpu")
    with torch.no_grad():
        out = trained(torch.randn(1, 8, 3), torch.randint(0, 2, (1, 8)).float(), torch.randn(1, 4, 3))
    assert torch.isfinite(out).all()


def test_problem_flag_contradicting_prior_raises(tmp_path):
    dump = tmp_path / "tiny_classification.h5"
    make_classification_dump(dump)
    prior = PriorDumpDataLoader(filename=str(dump), num_steps=2, batch_size=4, device="cpu")

    with pytest.raises(ValueError, match="the prior says"):
        pretrainTFM(prior=prior, problem="regression", device="cpu")


def test_problem_flag_rejects_unknown_values():
    with pytest.raises(ValueError, match="must be one of"):
        pretrainTFM(problem="clustering", device="cpu")


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
    prior = PriorDumpDataLoader(filename=str(dump), num_steps=2, batch_size=4, device="cpu")

    model = make_tiny_model(num_outputs=8)
    criterion = infer_criterion(model, prior, device="cpu")
    assert criterion.borders.shape == (9,)

    trained = pretrainTFM(model=model, prior=prior, eval=[], epochs=2, device="cpu")

    regressor = TabularRegressor(trained, dist=criterion, device="cpu")
    rng = np.random.default_rng(0)
    regressor.fit(rng.standard_normal((20, 3)), rng.standard_normal(20))
    predictions = regressor.predict(rng.standard_normal((5, 3)))
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
