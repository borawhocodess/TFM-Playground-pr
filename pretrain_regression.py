import argparse
import os
import uuid

import torch

from datetime import datetime
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION, TABARENA_TASKS
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import (PriorDumpDataLoader, LiveRegressionPriorDataLoader,
                                  make_bucket_edges_from_live_prior, CustomRegressionPriorDataLoader,
                                  make_bucket_edges_from_custom_prior)
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed, make_global_bucket_edges

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()

# data source
parser.add_argument("--priordump", type=str, default=None)
parser.add_argument("--live", action="store_true")
parser.add_argument("--prior", type=str, default="ticl_mlp", choices=["ticl_mlp", "custom"],
                    help="which live prior to use: 'ticl_mlp' (default TICL) or 'custom' (TabPFN-style bag)")

# live prior / curriculum args
parser.add_argument("--max_features", type=int, default=100)
parser.add_argument("--min_features", type=int, default=1)
parser.add_argument("--max_rows", type=int, default=1000)
parser.add_argument("--min_rows", type=int, default=50)
parser.add_argument("--curriculum_epochs", type=int, default=0)

# model
parser.add_argument("--heads", type=int, default=6)
parser.add_argument("--embeddingsize", type=int, default=192)
parser.add_argument("--hiddensize", type=int, default=768)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument("--residual_decay", type=float, default=1.0,
                    help="exponential residual decay per layer (1.0 = no decay)")
parser.add_argument("--thinking_rows", type=int, default=0,
                    help="number of learnable thinking rows prepended to the data (0 = disabled)")

# training
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--accumulate", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--loadcheckpoint", type=str, default=None)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--optimizer", type=str, default="schedulefree", choices=["schedulefree", "muon"])
parser.add_argument("--muon_lr", type=float, default=0.02)
parser.add_argument("--n_buckets", type=int, default=100)
parser.add_argument("--experiments-dir", type=str, default='workdir/experiments/regression')
parser.add_argument("--name", type=str, default='test')

args = parser.parse_args()

use_live = args.live or args.priordump is None

ts = datetime.now().strftime("%y%m%d-%H%M%S")
uid = uuid.uuid4().hex[:8]
e_name = args.name.strip()
e_id = f"{ts}-{uid}-{e_name}"
e_root = os.path.join(args.experiments_dir, e_name)
e_dir = os.path.join(e_root, e_id)
os.makedirs(e_dir, exist_ok=True)
print(f"experiment id: {e_id}")

set_randomness_seed(2402)

device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

if use_live:
    curriculum_epochs = args.curriculum_epochs if args.curriculum_epochs > 0 else 1
    _min_features = args.min_features if args.curriculum_epochs > 0 else args.max_features
    _min_rows = args.min_rows if args.curriculum_epochs > 0 else args.max_rows
    if args.prior == "custom":
        prior = CustomRegressionPriorDataLoader(
            num_steps=args.steps,
            batch_size=args.batchsize,
            max_features=args.max_features,
            max_rows=args.max_rows,
            min_features=_min_features,
            min_rows=_min_rows,
            device=device,
        )
        print("Computing bucket edges from custom prior sample...")
        bucket_edges = make_bucket_edges_from_custom_prior(
            n_buckets=args.n_buckets,
            batch_size=args.batchsize,
            num_features=args.max_features,
            num_rows=args.max_rows,
            device=device,
            n_batches=50,
        )
    else:
        prior = LiveRegressionPriorDataLoader(
            num_steps=args.steps,
            batch_size=args.batchsize,
            max_features=args.max_features,
            max_rows=args.max_rows,
            min_features=_min_features,
            min_rows=_min_rows,
            device=device,
        )
        print("Computing bucket edges from live prior sample...")
        bucket_edges = make_bucket_edges_from_live_prior(
            prior=prior._prior,
            n_buckets=args.n_buckets,
            batch_size=args.batchsize,
            num_features=args.max_features,
            num_rows=args.max_rows,
            device=device,
            n_batches=50,
        )
else:
    prior = PriorDumpDataLoader(
        filename=args.priordump,
        num_steps=args.steps,
        batch_size=args.batchsize,
        device=device,
        starting_index=args.steps * (ckpt['epoch'] if ckpt else 0),
    )
    bucket_edges = make_global_bucket_edges(
        filename=args.priordump,
        n_buckets=args.n_buckets,
        device=device,
    )

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
    residual_decay=args.residual_decay,
    num_thinking_rows=args.thinking_rows,
)

if ckpt:
    model.load_state_dict(ckpt['model'])

# bake bucket borders into the model so the checkpoint is self-contained
model.borders = bucket_edges

dist = FullSupportBarDistribution(bucket_edges)


class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks, max_samples=10_000, max_features=500):
        self.tasks = tasks
        self.max_samples = max_samples
        self.max_features = max_features

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, diag: str = "", **kwargs):
        regressor = NanoTabPFNRegressor(model, device=device)
        predictions = get_openml_predictions(
            model=regressor,
            tasks=self.tasks,
            max_n_samples=self.max_samples,
            max_n_features=self.max_features,
            classification=False,
        )
        if not predictions:
            print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | loss {loss:5.2f} | {diag} | no regression tasks',
                  flush=True)
            return
        scores = {name: r2_score(y_true, y_pred) for name, (y_true, y_pred, _) in predictions.items()}
        avg_score = sum(scores.values()) / len(scores)
        per_dataset = '  '.join(f'{name.lower().split("_")[0].split("-")[0]}={r2:.3f}' for name, r2 in scores.items())
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | loss {loss:5.2f} | {diag} | '
              f'avg r2 {avg_score:.3f} | {per_dataset}',
              flush=True)


callbacks = [EvaluationLoggerCallback(TABARENA_TASKS)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt,
    run_name=e_name,
    experiment_id=e_id,
    experiment_dir=e_dir,
    warmup_steps=args.warmup_steps,
    optimizer_type=args.optimizer,
    muon_lr=args.muon_lr,
)
