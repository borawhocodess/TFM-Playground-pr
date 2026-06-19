from types import SimpleNamespace

from sklearn.metrics import roc_auc_score
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback, WandbLoggerCallback
from tfmplayground.evaluation import TABARENA_TASKS, TOY_TASKS_CLASSIFICATION, get_openml_predictions
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.interface import TabularClassifier
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, load_config, set_randomness_seed

args = SimpleNamespace(**load_config("nanotabpfn_classifier"))

set_randomness_seed(2402)

device = get_default_device()

prior = PriorDumpDataLoader(
    filename=args.priordump,
    num_steps=args.steps,
    batch_size=args.batchsize,
    device=device,
)

criterion = nn.CrossEntropyLoss()

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=prior.max_num_classes,
)


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
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    multi_gpu=args.multigpu,
)
