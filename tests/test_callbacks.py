from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from tfmplayground import pretrainTFM
from tfmplayground.callbacks import Callback, ConsoleLoggerCallback, ExperimentCallback, safe_name
from tfmplayground.models import NanoTabPFNModel
from tfmplayground.priors import FunctionPrior


class Measuring(Callback):
    """Stands in for the evaluation callback, which is the thing that actually measures."""

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        return {"score": 0.5 + epoch}

    def close(self):
        pass


class Recording(Callback):
    """Stands in for a logger, and keeps what it was handed so a test can look at it."""

    def __init__(self):
        self.seen = []

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.seen.append(kwargs)

    def close(self):
        pass


def get_batch(batch_size, num_datapoints, num_features):
    x = torch.randn(batch_size, num_datapoints, num_features)
    y = (x[:, :, 0] > 0).float()
    sep = num_datapoints // 2
    return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


def train_with(callbacks, epochs=2):
    torch.manual_seed(0)
    return pretrainTFM(
        model=NanoTabPFNModel(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=2
        ),
        prior=FunctionPrior(get_batch_function=get_batch, num_datapoints_max=16, num_features=4, device="cpu"),
        eval=callbacks,
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
        steps_per_epoch=1,
        batch_size=2,
        device="cpu",
    )


def test_measurements_reach_later_callbacks():
    """Without this the eval score only ever reaches the console, and no logger can record it."""
    recorder = Recording()
    train_with([Measuring(), recorder])

    assert [seen["score"] for seen in recorder.seen] == [1.5, 2.5]


def test_measurements_do_not_reach_earlier_callbacks():
    """Order is the contract: a logger placed before the measurement gets nothing to log."""
    recorder = Recording()
    train_with([recorder, Measuring()])

    assert all("score" not in seen for seen in recorder.seen)


def test_console_logger_reports_what_was_measured(capsys):
    """The console line grows to carry whatever the callbacks before it measured."""
    train_with([Measuring(), ConsoleLoggerCallback()], epochs=1)

    printed = capsys.readouterr().out
    assert "Mean Loss" in printed
    assert "score 1.500" in printed


@dataclass
class Settings:
    seed: int = 11
    lr: float = 3e-4


def test_experiment_callback_writes_one_record(tmp_path):
    """The run directory has to answer what the run was without reference to anything else."""
    callback = ExperimentCallback(
        name="unit", experiments_dir=str(tmp_path), config=Settings(), console=False, source=""
    )
    train_with([Measuring(), callback], epochs=2)
    callback.close()

    record = (tmp_path / "classification" / "unit" / callback.e_id / f"{callback.e_id}-log.txt").read_text()
    assert "host:" in record and "torch:" in record  # environment
    assert "seed: 11" in record and "lr: 0.0003" in record  # config
    assert "params:" in record and "NanoTabPFNModel" in record  # what was trained
    assert "e:1 l:" in record and "score:1.5000" in record  # per epoch, with the measurement
    assert "runtime:" in record and f"experiment done: {callback.e_id}" in record  # footer


def test_experiment_ids_carry_the_name_and_do_not_collide(tmp_path):
    """Two runs started in the same second still get their own directory."""
    first = ExperimentCallback(name="unit", experiments_dir=str(tmp_path), console=False, source="")
    second = ExperimentCallback(name="unit", experiments_dir=str(tmp_path), console=False, source="")

    assert first.e_id != second.e_id
    assert first.e_id.endswith("-unit") and second.e_id.endswith("-unit")
    assert len(list((tmp_path / "classification" / "unit").iterdir())) == 2


def test_experiments_are_grouped_by_problem(tmp_path):
    """Classification and regression runs are not comparable, so they do not share a folder."""
    classifier = ExperimentCallback(name="unit", experiments_dir=str(tmp_path), console=False, source="")
    regressor = ExperimentCallback(
        name="unit", problem="regression", experiments_dir=str(tmp_path), console=False, source=""
    )

    assert (tmp_path / "classification" / "unit" / classifier.e_id).is_dir()
    assert (tmp_path / "regression" / "unit" / regressor.e_id).is_dir()
    assert (
        "problem: regression"
        in (tmp_path / "regression" / "unit" / regressor.e_id / f"{regressor.e_id}-log.txt").read_text()
    )


def test_experiment_callback_rejects_an_unknown_problem(tmp_path):
    """A typo would quietly start a third folder that nothing else knows about."""
    with pytest.raises(ValueError, match="problem must be one of"):
        ExperimentCallback(problem="ranking", experiments_dir=str(tmp_path), source="")


def test_experiment_callback_survives_an_unreadable_source(tmp_path):
    """sys.argv[0] is whatever launched us, and under pytest that is a binary."""
    binary = tmp_path / "not-text"
    binary.write_bytes(b"\x00\x01\x02\xff")

    callback = ExperimentCallback(name="", experiments_dir=str(tmp_path), console=False, source=str(binary))
    callback.close()
    record = tmp_path / "classification" / callback.e_id / f"{callback.e_id}-log.txt"
    assert "experiment done:" in record.read_text()


def test_callback_measurement_is_optional():
    """A callback that only reports returns nothing, and that must not break the chain."""
    recorder = Recording()
    train_with([ConsoleLoggerCallback(), recorder], epochs=1)

    assert len(recorder.seen) == 1


class Exploding(Callback):
    """Stands in for anything that can fail mid training."""

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        raise RuntimeError("boom")

    def close(self):
        pass


def test_a_crashed_run_is_not_recorded_as_finished(tmp_path):
    """A record that claims success when the run died is worse than no record, because it is believed."""
    callback = ExperimentCallback(name="unit", experiments_dir=str(tmp_path), console=False, source="")
    with pytest.raises(RuntimeError, match="boom"):
        train_with([Exploding(), callback], epochs=1)

    record = (tmp_path / "classification" / "unit" / callback.e_id / f"{callback.e_id}-log.txt").read_text()
    assert "status: failed: RuntimeError: boom" in record
    assert callback.status.startswith("failed")


def test_a_completed_run_says_so(tmp_path):
    callback = ExperimentCallback(name="unit", experiments_dir=str(tmp_path), console=False, source="")
    train_with([callback], epochs=1)
    callback.close()

    record = (tmp_path / "classification" / "unit" / callback.e_id / f"{callback.e_id}-log.txt").read_text()
    assert "status: completed" in record


@pytest.mark.parametrize(
    "name, expected",
    [
        ("../../etc", "etc"),
        ("a/b", "a-b"),
        (".hidden", "hidden"),
        ("keeps_this-1.0", "keeps_this-1.0"),
        ("  spaced  out ", "spaced-out"),
    ],
)
def test_names_cannot_escape_the_experiments_directory(name, expected):
    """Names go straight into paths, so a slash or a .. would write somewhere else entirely."""
    assert safe_name(name) == expected


def test_a_name_that_traverses_stays_inside(tmp_path):
    callback = ExperimentCallback(name="../escape", experiments_dir=str(tmp_path), console=False, source="")
    assert Path(callback.e_dir).resolve().is_relative_to(Path(tmp_path).resolve())


def test_a_callback_cannot_shadow_the_criterion():
    """dist is the training loop's own keyword, and a duplicate would be a TypeError."""

    class ShadowsDist(Callback):
        def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
            return {"dist": "not the criterion"}

        def close(self):
            pass

    recorder = Recording()
    train_with([ShadowsDist(), recorder], epochs=1)
    assert recorder.seen[0]["dist"] is not None and recorder.seen[0]["dist"] != "not the criterion"


def test_a_measurement_that_is_not_a_mapping_is_refused():
    """`if reported:` used to accept a float and then explode inside dict.update()."""

    class ReturnsAFloat(Callback):
        def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
            return 0.5

        def close(self):
            pass

    with pytest.raises(TypeError, match="have to be a mapping"):
        train_with([ReturnsAFloat()], epochs=1)
