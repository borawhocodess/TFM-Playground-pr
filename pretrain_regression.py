from dataclasses import asdict

from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.configs.models import NanoTabPFNRegressorConfig
from tfmplayground.configs.priors import RegressionPriorDumpConfig
from tfmplayground.configs.training import TrainingConfig
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, get_openml_predictions
from tfmplayground.interface import TabularRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, make_global_bucket_edges, set_randomness_seed

dump_config = RegressionPriorDumpConfig()
training_config = TrainingConfig()

set_randomness_seed(2402)

device = get_default_device()

prior = PriorDumpDataLoader(
    filename=dump_config.filename,
    num_steps=training_config.steps,
    batch_size=training_config.batch_size,
    device=device,
)

model_config = NanoTabPFNRegressorConfig()

model = NanoTabPFNModel(**asdict(model_config))

bucket_edges = make_global_bucket_edges(
    filename=dump_config.filename,
    n_buckets=model_config.num_outputs,
    device=device,
)

dist = FullSupportBarDistribution(bucket_edges)


class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        regressor = TabularRegressor(model, dist, device)
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
    criterion=dist,
    epochs=training_config.epochs,
    lr=training_config.lr,
    device=device,
    callbacks=callbacks,
)
