from pathlib import Path

import numpy as np
import pytest
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground.configs.models import NanoTabPFNClassifierConfig, NanoTabPFNRegressorConfig
from tfmplayground.configs.training import ClassificationExperimentConfig
from tfmplayground.interface import TabularRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import Prior
from tfmplayground.training.callbacks import Callback
from tfmplayground.training.train import train
from tfmplayground.utils import Experiment, QuantileLoss, load_model


class ClassificationPrior(Prior):
    def batch(self, batch_size):
        x = torch.randn(batch_size, 20, 3)
        y = torch.randint(0, 3, (batch_size, 20)).float()
        return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]


class Scores(Callback):
    def __init__(self, experiment, scores):
        self.experiment = experiment
        self.scores = list(scores)

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.experiment.score = self.scores.pop(0)

    def close(self):
        pass


class Quiet(Callback):
    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        pass

    def close(self):
        pass


def small(outputs):
    return dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=outputs)


def model():
    return NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3)))


def only(tmp_path, kind):
    return next(tmp_path.rglob(f"*-ckpt-{kind}.pth"))


def read(tmp_path, kind):
    return torch.load(only(tmp_path, kind), map_location="cpu", weights_only=False)


def experiment(tmp_path):
    return Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))


def run(tmp_path, scores=None, epochs=1, path=True):
    run_experiment = experiment(tmp_path) if path else None
    callbacks = [Scores(run_experiment, scores)] if scores else [Quiet()]
    return train(
        model=model(),
        prior=ClassificationPrior(),
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
        batch_size=2,
        steps_per_epoch=1,
        lr=1e-4,
        grad_clip=1.0,
        device="cpu",
        callbacks=callbacks,
        experiment=run_experiment,
    )


def test_a_run_with_a_score_writes_two_files(tmp_path):
    run(tmp_path, [0.1, 0.2], epochs=2)
    assert sorted(p.name.split("-ckpt-")[1] for p in tmp_path.rglob("*.pth")) == ["best.pth", "last.pth"]


def test_a_run_with_no_score_writes_the_last_file_only(tmp_path):
    run(tmp_path, epochs=2)
    assert [p.name.split("-ckpt-")[1] for p in tmp_path.rglob("*.pth")] == ["last.pth"]


def test_best_is_not_rewritten_when_the_score_drops(tmp_path):
    run(tmp_path, [0.9, 0.1, 0.2], epochs=3)
    assert only(tmp_path, "best").stat().st_mtime_ns < only(tmp_path, "last").stat().st_mtime_ns


def test_no_path_writes_nothing(tmp_path):
    run(tmp_path, path=False)
    assert list(tmp_path.rglob("*.pth")) == []


def test_the_file_carries_what_rebuilds_the_model(tmp_path):
    trained, _ = run(tmp_path)
    checkpoint = read(tmp_path, "last")
    assert checkpoint["model_class"] == "NanoTabPFNModel"
    assert checkpoint["config_class"] == "NanoTabPFNClassifierConfig"
    assert checkpoint["model_config"]["num_outputs"] == 3
    for a, b in zip(checkpoint["model_state"].values(), trained.to("cpu").state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_the_file_names_its_run_and_version(tmp_path):
    run_experiment = experiment(tmp_path)
    saved = model()
    run_experiment.save_checkpoint(run_experiment.last_checkpoint_path, saved)
    checkpoint = read(tmp_path, "last")
    assert checkpoint["experiment_id"] == run_experiment.id
    assert checkpoint["version"]
    assert checkpoint["problem"] == "classification"


def test_the_file_holds_the_model_only(tmp_path):
    run(tmp_path, [0.1, 0.2], epochs=2)
    for kind in ("best", "last"):
        assert set(read(tmp_path, kind)) == {
            "version",
            "experiment_id",
            "problem",
            "model_class",
            "config_class",
            "model_config",
            "model_state",
        }


def test_the_loaded_model_matches_the_trained_one(tmp_path):
    trained, _ = run(tmp_path)
    loaded = load_model(only(tmp_path, "last"))
    assert type(loaded) is type(trained)
    assert loaded.config == trained.config
    for a, b in zip(loaded.state_dict().values(), trained.to("cpu").state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_the_loaded_borders_come_back_fitted(tmp_path):
    run_experiment = experiment(tmp_path)
    saved = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9)))
    saved.borders = torch.linspace(-3, 3, 10)
    run_experiment.save_checkpoint(run_experiment.last_checkpoint_path, saved)
    loaded = load_model(only(tmp_path, "last"))
    assert torch.equal(loaded.borders, saved.borders)
    assert isinstance(TabularRegressor(loaded, device="cpu").dist, FullSupportBarDistribution)


def test_a_loaded_regressor_predicts_without_anything_else(tmp_path):
    config = NanoTabPFNRegressorConfig(**small(9))
    config.head = "quantiles"
    train(
        model=NanoTabPFNModel(config=config),
        prior=ClassificationPrior(),
        criterion=QuantileLoss(9),
        epochs=1,
        batch_size=2,
        steps_per_epoch=1,
        lr=1e-4,
        grad_clip=1.0,
        device="cpu",
        callbacks=[Quiet()],
        experiment=experiment(tmp_path),
    )
    regressor = TabularRegressor(load_model(only(tmp_path, "last")), device="cpu")
    assert isinstance(regressor.dist, QuantileLoss)
    regressor.fit(np.zeros((8, 3)), np.arange(8, dtype=float))
    assert regressor.predict(np.zeros((4, 3))).shape == (4,)


def test_a_regression_file_names_its_own_problem(tmp_path):
    run_experiment = experiment(tmp_path)
    saved = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9)))
    run_experiment.save_checkpoint(run_experiment.last_checkpoint_path, saved)
    assert read(tmp_path, "last")["problem"] == "regression"


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    run(tmp_path, epochs=1)
    target = only(tmp_path, "last")
    good = target.read_bytes()
    run_experiment = experiment(tmp_path)
    saved = model()

    def half_write(checkpoint, path):
        Path(path).write_bytes(b"half a file")
        raise OSError("the disk is full")

    monkeypatch.setattr(torch, "save", half_write)
    with pytest.raises(OSError):
        run_experiment.save_checkpoint(target, saved)
    assert target.read_bytes() == good


def test_a_score_that_is_not_finite_never_becomes_best(tmp_path):
    run_experiment = experiment(tmp_path)
    saved = model()
    run_experiment.score = float("nan")
    run_experiment.save_checkpoints(saved)
    assert not run_experiment.best_checkpoint_path.exists()
    run_experiment.score = 0.5
    run_experiment.save_checkpoints(saved)
    assert run_experiment.best_score == 0.5
