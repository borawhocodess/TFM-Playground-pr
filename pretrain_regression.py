from types import SimpleNamespace

from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, get_openml_predictions
from tfmplayground.interface import TabularRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, load_config, make_global_bucket_edges, set_randomness_seed

args = SimpleNamespace(**load_config("nanotabpfn_regressor"))

set_randomness_seed(2402)

device = get_default_device()

prior = PriorDumpDataLoader(
    filename=args.priordump,
    num_steps=args.steps,
    batch_size=args.batchsize,
    device=device,
)

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

bucket_edges = make_global_bucket_edges(
    filename=args.priordump,
    n_buckets=args.n_buckets,
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
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
)
