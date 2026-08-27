import argparse

from torch import nn

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import TabICLClassifierConfig
from tfmplayground.configs.priors import TabICLClassificationPriorConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, TrainingConfig
from tfmplayground.evaluation.evaluation import TABARENA_TASKS, TOY_TASKS_CLASSIFICATION
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import TabICLPrior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.train import train
from tfmplayground.utils import Experiment, get_default_device, set_randomness_seed

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="test")
parser.add_argument("--tasks", default="toy", choices=["toy", "tabarena"])
args = parser.parse_args()

tasks = TOY_TASKS_CLASSIFICATION if args.tasks == "toy" else TABARENA_TASKS

prior_config = TabICLClassificationPriorConfig(num_datapoints_max=256, num_features_max=4)
training_config = TrainingConfig()

set_randomness_seed(training_config.seed)

device = get_default_device()

prior = TabICLPrior(config=prior_config, device=device)

criterion = nn.CrossEntropyLoss()

model_config = TabICLClassifierConfig()

model = TabICLModel(config=model_config)


experiment_config = ClassificationExperimentConfig(name=args.name)

experiment = Experiment(config=experiment_config)

evaluation_config = EvaluationConfig()

callbacks = [ClassifierExperimentEvaluationCallback(experiment, config=evaluation_config, tasks=tasks, device=device)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=training_config.epochs,
    batch_size=training_config.batch_size,
    steps_per_epoch=training_config.steps,
    lr=training_config.lr,
    grad_clip=training_config.grad_clip,
    device=device,
    callbacks=callbacks,
)
