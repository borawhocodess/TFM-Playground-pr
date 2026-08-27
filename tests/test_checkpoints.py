import schedulefree
import torch
from torch import nn

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import Prior
from tfmplayground.training.callbacks import Callback
from tfmplayground.training.train import train
from tfmplayground.utils import Experiment


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


def optimizer(model):
    return schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-4)


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


def test_last_keeps_the_final_epoch(tmp_path):
    run(tmp_path, epochs=3)
    assert read(tmp_path, "last")["epoch"] == 3


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
    run_experiment.save_checkpoint(run_experiment.last_checkpoint_path, saved, optimizer(saved), 1, None)
    checkpoint = read(tmp_path, "last")
    assert checkpoint["experiment_id"] == run_experiment.id
    assert checkpoint["version"]


def test_both_files_hold_the_run_state(tmp_path):
    run(tmp_path, [0.1, 0.2], epochs=2)
    for kind in ("best", "last"):
        checkpoint = read(tmp_path, kind)
        assert checkpoint["optimizer_state"]["state"] is not None
        assert set(checkpoint["random_state"]) == {"python", "numpy", "torch"}
        assert "prior_pointer" in checkpoint
