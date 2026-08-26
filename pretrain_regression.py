from sklearn.metrics import r2_score

from tfmplayground.configs.models import TabICLRegressorConfig
from tfmplayground.configs.priors import TabICLRegressionPriorConfig
from tfmplayground.configs.training import TrainingConfig
from tfmplayground.evaluation.evaluation import TOY_TASKS_REGRESSION, get_openml_predictions
from tfmplayground.interface import TabularRegressor
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import TabICLPrior
from tfmplayground.training.callbacks import ConsoleLoggerCallback
from tfmplayground.training.train import train
from tfmplayground.utils import QuantileLoss, get_default_device, set_randomness_seed

prior_config = TabICLRegressionPriorConfig(num_datapoints_max=400, num_features_max=20)
training_config = TrainingConfig()

set_randomness_seed(training_config.seed)

device = get_default_device()

prior = TabICLPrior(config=prior_config, device=device)

model_config = TabICLRegressorConfig()

model = TabICLModel(config=model_config)

criterion = QuantileLoss(n_quantiles=model_config.num_quantiles)


class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        regressor = TabularRegressor(model, criterion, device)
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for _dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}",
            flush=True,
        )


callbacks = [EvaluationLoggerCallback(TOY_TASKS_REGRESSION)]

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
