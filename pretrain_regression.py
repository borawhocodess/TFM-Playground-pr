import argparse

from pfns.bar_distribution import FullSupportBarDistribution

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import NanoTabPFNRegressorConfig
from tfmplayground.configs.priors import TabICLRegressionPriorConfig
from tfmplayground.configs.training import RegressionExperimentConfig, TrainingConfig
from tfmplayground.evaluation.evaluation import TABARENA_TASKS, TOY_TASKS_REGRESSION
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import TabICLPrior
from tfmplayground.training.callbacks import RegressorExperimentEvaluationCallback
from tfmplayground.training.train import train
from tfmplayground.utils import Experiment, get_default_device, make_bucket_borders, set_randomness_seed

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="test")
parser.add_argument("--tasks", default="toy", choices=["toy", "tabarena"])
args = parser.parse_args()

tasks = TOY_TASKS_REGRESSION if args.tasks == "toy" else TABARENA_TASKS

prior_config = TabICLRegressionPriorConfig(num_datapoints_max=256, num_features_max=4)
model_config = NanoTabPFNRegressorConfig()
training_config = TrainingConfig()
experiment_config = RegressionExperimentConfig(name=args.name)
evaluation_config = EvaluationConfig()

experiment = Experiment(config=experiment_config)

set_randomness_seed(training_config.seed)

device = get_default_device()

prior = TabICLPrior(config=prior_config, device=device)

model = NanoTabPFNModel(config=model_config)

model.borders = make_bucket_borders(
    prior=prior,
    num_buckets=model_config.num_outputs,
    batch_size=training_config.batch_size,
    min_targets=training_config.bucket_borders_min_targets,
    outlier_threshold=training_config.bucket_borders_outlier_threshold,
).to(device)

criterion = FullSupportBarDistribution(model.borders).to(device)

callbacks = [RegressorExperimentEvaluationCallback(experiment, config=evaluation_config, tasks=tasks, device=device)]

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
