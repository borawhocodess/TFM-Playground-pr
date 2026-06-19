import argparse

from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, get_openml_predictions
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, make_global_bucket_edges, set_randomness_seed

parser = argparse.ArgumentParser()

parser.add_argument("--priordump", type=str, default="50x3_1280k_regression.h5", help="path to the prior dump")
parser.add_argument("--heads", type=int, default=6, help="number of attention heads")
parser.add_argument("--embeddingsize", type=int, default=192, help="the size of the embeddings used for the cells")
parser.add_argument("--hiddensize", type=int, default=768, help="size of the hidden layer of the mlps")
parser.add_argument("--layers", type=int, default=6, help="number of transformer layers")
parser.add_argument(
    "--batchsize", type=int, default=1, help="batch size used during training (before gradient accumulation)"
)
parser.add_argument(
    "--accumulate", type=int, default=1, help="number of gradients to accumulate before updating the weights"
)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--steps", type=int, default=100, help="number of steps that constitute one epoch (important for lr scheduler)"
)
parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--n_buckets", type=int, default=100, help="number of buckets for the data loader")

args = parser.parse_args()

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
        regressor = NanoTabPFNRegressor(model, dist, device)
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
