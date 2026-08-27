from torch import nn

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import TabICLClassifierConfig, TabICLRegressorConfig
from tfmplayground.configs.priors import TabICLClassificationPriorConfig, TabICLRegressionPriorConfig
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ClassificationTrainingConfig,
    RegressionExperimentConfig,
    RegressionTrainingConfig,
)
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import TabICLPrior
from tfmplayground.training.callbacks import (
    ClassifierExperimentEvaluationCallback,
    RegressorExperimentEvaluationCallback,
)
from tfmplayground.training.train import train
from tfmplayground.utils import (
    Experiment,
    get_default_device,
    make_bucket_borders,
    make_regression_decoder,
    set_randomness_seed,
)


def default_training(problem):
    if problem == "classification":
        return ClassificationTrainingConfig()
    if problem == "regression":
        return RegressionTrainingConfig()


def default_experiment(problem):
    if problem == "classification":
        return Experiment(config=ClassificationExperimentConfig())
    if problem == "regression":
        return Experiment(config=RegressionExperimentConfig())


def default_prior(problem, device):
    if problem == "classification":
        return TabICLPrior(config=TabICLClassificationPriorConfig(), device=device)
    if problem == "regression":
        return TabICLPrior(config=TabICLRegressionPriorConfig(), device=device)


def default_model(problem):
    if problem == "classification":
        return TabICLModel(config=TabICLClassifierConfig())
    if problem == "regression":
        return TabICLModel(config=TabICLRegressorConfig())


def default_callback(problem, experiment, config, device):
    if problem == "classification":
        return ClassifierExperimentEvaluationCallback(experiment, config=config, device=device)
    if problem == "regression":
        return RegressorExperimentEvaluationCallback(experiment, config=config, device=device)


def check_problems(problem, model, prior, training):
    if problem not in ("classification", "regression"):
        raise ValueError(f"the problem must be classification or regression, not {problem!r}")
    for x in (model, prior, training):
        x_config = getattr(x, "config", x)
        x_problem = getattr(x_config, "problem", None)
        if x_problem is not None and x_problem != problem:
            raise ValueError(f"{type(x).__name__} is built for {x_problem}, not {problem}")


def default_criterion(problem, model, prior, training, device):
    if problem == "classification":
        return nn.CrossEntropyLoss()
    if problem == "regression":
        head = training.criterion if training.criterion is not None else model.config.head
        if head == "buckets":
            model.borders = make_bucket_borders(
                prior=prior,
                num_buckets=model.borders.numel() - 1,
                batch_size=training.batch_size,
                min_targets=training.bucket_borders_min_targets,
                outlier_threshold=training.bucket_borders_outlier_threshold,
            ).to(device)
        model.config.head = head
        return make_regression_decoder(model).to(device)


def pretrainTFM(problem, model=None, prior=None, eval=None, training=None, device=None):
    check_problems(problem, model, prior, training)
    experiment = default_experiment(problem)
    training = training if training is not None else default_training(problem)
    set_randomness_seed(training.seed)
    device = device if device is not None else get_default_device()
    prior = prior if prior is not None else default_prior(problem, device)
    model = model if model is not None else default_model(problem)
    eval = eval if eval is not None else EvaluationConfig()
    criterion = default_criterion(problem, model, prior, training, device)
    callback = default_callback(problem, experiment, eval, device)

    trained_model, _ = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=training.epochs,
        batch_size=training.batch_size,
        steps_per_epoch=training.steps,
        lr=training.lr,
        grad_clip=training.grad_clip,
        device=device,
        callbacks=[callback],
    )
    return trained_model
