from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, RegressionExperimentConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.training.callbacks import ExperimentCallback
from tfmplayground.utils import Experiment


def read(experiment):
    return experiment.log_path.read_text()


def test_the_run_makes_its_own_directory(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    assert experiment.dir.is_dir()
    assert experiment.dir.name == experiment.id


def test_the_run_names_its_own_files(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    assert experiment.log_path == experiment.dir / f"{experiment.id}-log.txt"
    assert experiment.best_checkpoint_path == experiment.dir / f"{experiment.id}-ckpt-best.pth"
    assert experiment.last_checkpoint_path == experiment.dir / f"{experiment.id}-ckpt-last.pth"


def test_a_run_that_was_never_named_goes_under_test(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    assert experiment.dir.parent.name == "test"
    assert experiment.id.endswith("-test")


def test_the_two_problems_never_share_a_directory(tmp_path):
    classification = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    regression = Experiment(config=RegressionExperimentConfig(experiments_dir=tmp_path))
    assert classification.dir.parent.parent.name == "classification"
    assert regression.dir.parent.parent.name == "regression"


def test_a_named_run_gets_a_folder_of_its_own(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path, name="tabicl"))
    assert experiment.dir.parent.name == "tabicl"
    assert experiment.id.endswith("-tabicl")


def test_two_runs_of_one_name_never_collide(tmp_path):
    first = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path, name="tabicl"))
    second = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path, name="tabicl"))
    assert first.id != second.id
    assert first.log_path != second.log_path


def test_a_name_wiped_on_purpose_carries_no_trailing_dash(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path, name=""))
    assert not experiment.id.endswith("-")
    assert experiment.dir.parent.name == "classification"


def test_runs_sort_by_the_time_they_started(tmp_path):
    first = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    second = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    assert first.id[:13] <= second.id[:13]


def test_every_line_reaches_the_log(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    experiment.print0("e:1 l:2.3040")
    experiment.print0("e:2 l:2.1010")
    assert read(experiment).splitlines() == ["e:1 l:2.3040", "e:2 l:2.1010"]


def test_a_line_reaches_the_console_only_when_asked(tmp_path, capsys):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    experiment.print0("quiet")
    experiment.print0("loud", console=True)
    printed = capsys.readouterr().out
    assert "quiet" not in printed
    assert "loud" in printed
    assert "quiet" in read(experiment)


def run_callback(tmp_path, epochs):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    callback = ExperimentCallback(experiment)
    for epoch in range(1, epochs + 1):
        callback.on_epoch_end(epoch, 12.5, 1.6659, None)
    callback.close()
    return experiment


def test_the_run_opens_with_its_own_name(tmp_path, capsys):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    ExperimentCallback(experiment)
    assert f"experiment: {experiment.id}" in capsys.readouterr().out
    assert f"experiment: {experiment.id}" in read(experiment)


def test_every_epoch_writes_one_line(tmp_path):
    experiment = run_callback(tmp_path, epochs=3)
    lines = [line for line in read(experiment).splitlines() if line.startswith("e:")]
    assert lines == [f"e:{epoch} l:1.6659 e_t:12.50s" for epoch in (1, 2, 3)]


def test_the_epoch_line_stays_off_the_console(tmp_path, capsys):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    callback = ExperimentCallback(experiment)
    capsys.readouterr()
    callback.on_epoch_end(1, 12.5, 1.6659, None)
    assert capsys.readouterr().out == ""
    assert "e:1" in read(experiment)


def test_the_log_closes_with_what_the_run_cost(tmp_path):
    text = read(run_callback(tmp_path, epochs=3))
    assert "runtime:" in text
    assert "mins" in text


def test_a_run_that_never_finished_an_epoch_still_closes(tmp_path):
    assert "runtime:" in read(run_callback(tmp_path, epochs=0))


def test_the_run_keeps_no_best_score_before_it_starts(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    assert experiment.best_score is None


def improves(experiment, model, score):
    before = experiment.best_checkpoint_path.stat().st_mtime_ns if experiment.best_checkpoint_path.exists() else None
    experiment.score = score
    experiment.save_checkpoints(model)
    after = experiment.best_checkpoint_path.stat().st_mtime_ns if experiment.best_checkpoint_path.exists() else None
    return before != after


def test_a_score_that_does_not_improve_writes_nothing(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=3
        )
    )
    assert improves(experiment, model, 0.5)
    assert not improves(experiment, model, 0.4)
    assert not improves(experiment, model, 0.5)
    assert improves(experiment, model, 0.6)
    assert experiment.best_score == 0.6


def test_no_score_writes_nothing(tmp_path):
    experiment = Experiment(config=ClassificationExperimentConfig(experiments_dir=tmp_path))
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=3
        )
    )
    assert not improves(experiment, model, None)
    assert not experiment.best_checkpoint_path.exists()
