import numpy as np
import pytest
import torch

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import (
    ModdedNanoTabPFNRegressorConfig,
    NanoTabICLRegressorConfig,
    NanoTabPFNClassifierConfig,
    NanoTabPFNRegressorConfig,
    TabFMRegressorConfig,
    TabICLRegressorConfig,
)
from tfmplayground.configs.priors import TabICLClassificationPriorConfig
from tfmplayground.configs.training import RegressionTrainingConfig, TrainingConfig
from tfmplayground.evaluation.evaluation import (
    TABARENA_TASKS,
    TOY_TASKS_CLASSIFICATION,
    TOY_TASKS_REGRESSION,
    task_ids,
)
from tfmplayground.models.nanotabicl import NanoTabICLModel
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import Prior, TabICLPrior
from tfmplayground.training import callbacks as callbacks_source
from tfmplayground.training import pretrain as pretrain_module
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM
from tfmplayground.utils import Experiment, QuantileLoss, ScalarMSELoss

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


class TinyPrior(Prior):
    def batch(self, batch_size):
        x = torch.randn(batch_size, 20, 3)
        y = torch.randint(0, 3, (batch_size, 20)).float()
        return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]


@pytest.fixture
def offline_openml(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=30)
    y_proba = rng.random((30, 3))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    predictions = {"toy": (y_true, y_proba.argmax(axis=1), y_proba)}
    monkeypatch.setattr(callbacks_source, "get_openml_predictions", lambda **kwargs: predictions)

    class ExperimentInTmp(Experiment):
        def __init__(self, config):
            config.experiments_dir = str(tmp_path)
            super().__init__(config)

    monkeypatch.setattr(pretrain_module, "Experiment", ExperimentInTmp)
    return predictions


def small(outputs):
    return dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=outputs)


def test_the_classification_call_trains_and_writes_its_run(tmp_path, offline_openml):
    model = pretrainTFM(
        problem="classification",
        model=NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3))),
        prior=TinyPrior(),
        eval=EvaluationConfig(tasks="toy"),
        training=TrainingConfig(epochs=2, steps=2, batch_size=2),
    )
    assert isinstance(model, NanoTabPFNModel)
    log = next(tmp_path.rglob("*-log.txt")).read_text()
    assert "e:1" in log and "e:2" in log
    assert "roc_auc:" in log
    assert "runtime:" in log


def test_the_regression_call_fits_borders_and_trains(tmp_path, offline_openml, monkeypatch):
    monkeypatch.setattr(
        pretrain_module, "make_bucket_borders", lambda **kwargs: torch.linspace(-3, 3, kwargs["num_buckets"] + 1)
    )
    model = pretrainTFM(
        problem="regression",
        model=NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9))),
        prior=TinyPrior(),
        eval=EvaluationConfig(tasks="toy"),
        training=RegressionTrainingConfig(epochs=1, steps=2, batch_size=2),
    )
    assert (model.borders.diff() > 0).all()
    assert "r2:" in next(tmp_path.rglob("*-log.txt")).read_text()


def test_the_bare_classification_call_builds_the_defaults(tmp_path, offline_openml, monkeypatch):
    seen = {}
    monkeypatch.setattr(pretrain_module, "train", lambda **kwargs: (seen.update(kwargs), (kwargs["model"], 0.0))[1])
    model = pretrainTFM(problem="classification")
    assert isinstance(model, TabICLModel)
    assert isinstance(seen["prior"], TabICLPrior)
    assert isinstance(seen["criterion"], torch.nn.CrossEntropyLoss)
    assert isinstance(seen["callbacks"][0], ClassifierExperimentEvaluationCallback)


def test_an_unknown_problem_is_refused():
    with pytest.raises(ValueError, match="classification or regression"):
        pretrainTFM(problem="clustering")


def test_the_seed_is_set_before_the_defaults_are_built(tmp_path, offline_openml, monkeypatch):
    monkeypatch.setattr(pretrain_module, "train", lambda **kwargs: (kwargs["model"], 0.0))
    first = pretrainTFM(problem="classification")
    torch.manual_seed(999)
    second = pretrainTFM(problem="classification")
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


def test_each_problem_reads_toy_as_its_own_list():
    assert task_ids("toy", "classification") == TOY_TASKS_CLASSIFICATION
    assert task_ids("toy", "regression") == TOY_TASKS_REGRESSION


def test_tabarena_is_named_not_listed():
    assert task_ids("tabarena", "classification") == TABARENA_TASKS
    assert task_ids("tabarena", "regression") == TABARENA_TASKS


def test_a_raw_list_of_tasks_still_works():
    assert task_ids([59, 2382], "classification") == [59, 2382]


def test_a_scalar_criterion_skips_the_borders(tmp_path, offline_openml, monkeypatch):
    monkeypatch.setattr(pretrain_module, "make_bucket_borders", lambda **kwargs: pytest.fail("borders were fitted"))
    model = pretrainTFM(
        problem="regression",
        model=NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(1))),
        prior=TinyPrior(),
        eval=EvaluationConfig(tasks="toy"),
        training=RegressionTrainingConfig(epochs=1, steps=2, batch_size=2, criterion="scalar"),
    )
    assert (model.borders == 0).all()
    assert "r2:" in next(tmp_path.rglob("*-log.txt")).read_text()


@pytest.mark.parametrize(
    ("config_class", "expected"),
    [
        (NanoTabPFNRegressorConfig, "buckets"),
        (ModdedNanoTabPFNRegressorConfig, "buckets"),
        (TabICLRegressorConfig, "quantiles"),
        (NanoTabICLRegressorConfig, "quantiles"),
        (TabFMRegressorConfig, "scalar"),
    ],
)
def test_each_regression_head_says_what_it_means(config_class, expected):
    assert config_class().head == expected


def test_a_quantile_head_gets_a_quantile_loss(tmp_path, offline_openml, monkeypatch):
    monkeypatch.setattr(pretrain_module, "train", lambda **kwargs: (seen.update(kwargs), (kwargs["model"], 0.0))[1])
    seen = {}
    pretrainTFM(
        problem="regression",
        model=NanoTabICLModel(config=NanoTabICLRegressorConfig(out_dim=9, **NANO)),
        prior=TinyPrior(),
        training=RegressionTrainingConfig(epochs=1, steps=1, batch_size=2),
    )
    assert isinstance(seen["criterion"], QuantileLoss)


def test_the_training_config_overrides_the_head(tmp_path, offline_openml, monkeypatch):
    seen = {}
    monkeypatch.setattr(pretrain_module, "train", lambda **kwargs: (seen.update(kwargs), (kwargs["model"], 0.0))[1])
    pretrainTFM(
        problem="regression",
        model=NanoTabICLModel(config=NanoTabICLRegressorConfig(out_dim=9, **NANO)),
        prior=TinyPrior(),
        training=RegressionTrainingConfig(epochs=1, steps=1, batch_size=2, criterion="scalar"),
    )
    assert isinstance(seen["criterion"], ScalarMSELoss)


def test_the_bare_regression_call_needs_no_borders(tmp_path, offline_openml, monkeypatch):
    seen = {}
    monkeypatch.setattr(pretrain_module, "train", lambda **kwargs: (seen.update(kwargs), (kwargs["model"], 0.0))[1])
    monkeypatch.setattr(pretrain_module, "make_bucket_borders", lambda **kwargs: pytest.fail("borders were fitted"))
    model = pretrainTFM(problem="regression")
    assert isinstance(model, TabICLModel)
    assert isinstance(seen["criterion"], QuantileLoss)


def test_the_readme_import_works():
    import tfmplayground

    assert tfmplayground.pretrainTFM is pretrainTFM
    assert set(tfmplayground.__all__) == {"TabularClassifier", "TabularRegressor", "pretrainTFM"}


def test_a_classifier_model_is_refused_for_regression(tmp_path, offline_openml):
    with pytest.raises(ValueError, match="built for classification"):
        pretrainTFM(
            problem="regression",
            model=NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3))),
            prior=TinyPrior(),
            training=RegressionTrainingConfig(epochs=1, steps=1, batch_size=2),
        )


def test_a_regressor_model_is_refused_for_classification(tmp_path, offline_openml):
    with pytest.raises(ValueError, match="built for regression"):
        pretrainTFM(
            problem="classification",
            model=NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9))),
            prior=TinyPrior(),
            training=TrainingConfig(epochs=1, steps=1, batch_size=2),
        )


def test_a_model_that_carries_no_config_is_let_through(tmp_path, offline_openml, monkeypatch):
    monkeypatch.setattr(pretrain_module, "train", lambda **kwargs: (kwargs["model"], 0.0))
    model = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9)))
    del model.config
    training = TrainingConfig(epochs=1, steps=1, batch_size=2)
    assert pretrainTFM(problem="classification", model=model, prior=TinyPrior(), training=training) is model


def test_a_classification_prior_is_refused_for_regression(tmp_path, offline_openml):
    with pytest.raises(ValueError, match="built for classification"):
        pretrainTFM(
            problem="regression",
            model=NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9))),
            prior=TabICLPrior(config=TabICLClassificationPriorConfig(), device="cpu"),
            training=RegressionTrainingConfig(epochs=1, steps=1, batch_size=2),
        )


def test_a_regression_training_config_is_refused_for_classification(tmp_path, offline_openml):
    with pytest.raises(ValueError, match="built for regression"):
        pretrainTFM(
            problem="classification",
            model=NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3))),
            prior=TinyPrior(),
            training=RegressionTrainingConfig(epochs=1, steps=1, batch_size=2),
        )
