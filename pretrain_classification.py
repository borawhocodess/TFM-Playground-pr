from sklearn.metrics import roc_auc_score
from torch import nn

from tfmplayground.configs.models import TabICLClassifierConfig
from tfmplayground.configs.priors import TabICLClassificationPriorConfig
from tfmplayground.configs.training import TrainingConfig
from tfmplayground.evaluation.evaluation import TABARENA_TASKS, TOY_TASKS_CLASSIFICATION, get_openml_predictions
from tfmplayground.interface import TabularClassifier
from tfmplayground.models.tabicl import TabICLModel
from tfmplayground.priors import TabICLPrior
from tfmplayground.training.callbacks import ConsoleLoggerCallback, WandbLoggerCallback
from tfmplayground.training.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

prior_config = TabICLClassificationPriorConfig(num_datapoints_max=50, num_features_max=3)
training_config = TrainingConfig()

set_randomness_seed(training_config.seed)

device = get_default_device()

prior = TabICLPrior(config=prior_config, device=device)

criterion = nn.CrossEntropyLoss()

model_config = TabICLClassifierConfig()

model = TabICLModel(config=model_config)


class ToyEvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = TabularClassifier(model, device)
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for _dataset_name, (y_true, _y_pred, y_proba) in predictions.items():
            scores.append(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        avg_score = sum(scores) / len(scores)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}",
            flush=True,
        )


class ProductionEvaluationLoggerCallback(WandbLoggerCallback):
    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        super().__init__(project, name, config, log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = TabularClassifier(model, device)
        predictions = get_openml_predictions(model=classifier, classification=True, tasks=TABARENA_TASKS)
        scores = []
        log_metrics = {"epoch": epoch, "epoch_time": epoch_time, "mean_loss": loss}
        for dataset_name, (y_true, _y_pred, y_proba) in predictions.items():
            score = roc_auc_score(y_true, y_proba, multi_class="ovr")
            scores.append(score)
            log_metrics[f"roc_auc/{dataset_name}"] = score
        avg_score = sum(scores) / len(scores)
        log_metrics["tabarena_avg_roc_auc"] = avg_score
        self.wandb.log(log_metrics)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg roc auc {avg_score:.3f}",
            flush=True,
        )


# callbacks = [ProductionEvaluationLoggerCallback('nanoTFM', args.runname)]
callbacks = [ToyEvaluationLoggerCallback(TOY_TASKS_CLASSIFICATION)]

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
