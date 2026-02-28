"""Hydra-based regression training entrypoint using BaseTrainer and PriorDumpDataset.

Accepts both Hydra overrides (e.g. training.epochs=1) and argparse-style flags
(e.g. --epochs 1 --steps 2). Flags are converted to Hydra overrides before Hydra runs.
"""

import argparse
import sys

import hydra
from omegaconf import DictConfig
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score
import torch

from tfmplayground.training.callbacks import ConsoleLoggerCallback
from tfmplayground.training.trainer import BaseTrainer
from tfmplayground.training.util import tqdm_on_main, find_latest_run_dir
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataset import PriorDumpDataset
from tfmplayground.utils import set_randomness_seed, make_global_bucket_edges

set_randomness_seed(2402)


def _argparse_to_hydra_argv(argv: list[str]) -> list[str]:
    """Convert argparse-style flags to Hydra overrides; return new argv for Hydra."""
    parser = argparse.ArgumentParser(description="Train regression (argparse flags → Hydra)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=None)
    parser.add_argument("--priordump", "--dataset", type=str, default=None, dest="priordump")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_buckets", type=int, default=None)
    parser.add_argument("--resume_dir", type=str, default=None, help="explicit run dir to resume from (else latest by mtime)")
    args, remaining = parser.parse_known_args(argv[1:])

    overrides = []
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.steps is not None:
        overrides.append(f"training.steps={args.steps}")
    if args.run_name is not None:
        overrides.append(f"training.run_name={args.run_name}")
    if args.resume is True:
        overrides.append("training.resume_from_checkpoint=true")
    if args.resume is False:
        overrides.append("training.resume_from_checkpoint=false")
    if args.priordump is not None:
        overrides.append(f"dataset.filename={args.priordump}")
    if args.batch_size is not None:
        overrides.append(f"dataset.batch_size={args.batch_size}")
    if args.lr is not None:
        overrides.append(f"training.lr={args.lr}")
    if args.n_buckets is not None:
        overrides.append(f"training.n_buckets={args.n_buckets}")
    if args.resume_dir is not None:
        overrides.append(f"training.resume_dir={args.resume_dir}")

    return [argv[0]] + overrides + remaining


class RegressionEvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks, device):
        self.tasks = tasks
        self.device = device

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        dist = kwargs.get("dist")
        if dist is None:
            return
        regressor = NanoTabPFNRegressor(model, dist, self.device)
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        tqdm_on_main(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}"
        )


@hydra.main(version_base=None, config_path="configs", config_name="train_regression")
def main(cfg: DictConfig):
    dataset = PriorDumpDataset(
        **dict(cfg.dataset),
        num_steps=cfg.training.steps,
    )
    device = str(cfg.dataset.device)
    n_buckets = int(cfg.training.n_buckets)
    bucket_edges = make_global_bucket_edges(
        filename=cfg.dataset.filename,
        n_buckets=n_buckets,
        device=device,
    )
    dist = FullSupportBarDistribution(bucket_edges)

    model = NanoTabPFNModel(
        **dict(cfg.model),
        num_outputs=n_buckets,
    )
    callbacks = [RegressionEvaluationLoggerCallback(TOY_TASKS_REGRESSION, device)]
    base_run_dir = "workdir/experiments/regression"
    run_name = cfg.training.run_name
    resume = cfg.training.get("resume_from_checkpoint", False)
    # When resuming: explicit training.resume_dir wins; else for named runs use latest by mtime
    resume_dir = None
    if resume:
        resume_dir = cfg.training.get("resume_dir") or (
            find_latest_run_dir(base_run_dir, run_name) if run_name else None
        )
    trainer = BaseTrainer(
        model=model,
        train_dataset=dataset,
        criterion=dist,
        callbacks=callbacks,
        initial_lr=cfg.training.lr,
        accumulate_gradients=cfg.training.accumulate_gradients,
        epochs=cfg.training.epochs,
        steps=cfg.training.steps,
        run_dir=base_run_dir,
        run_name=run_name,
        task="regression",
        resume_dir=resume_dir,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
    )
    raw_model = trainer.train(resume_from_checkpoint=resume)

    # Save bucket edges and final model weights to run_dir
    torch.save(bucket_edges, f"{trainer.run_dir}/bucket_edges.pth")
    torch.save(raw_model.to("cpu").state_dict(), f"{trainer.run_dir}/final_model.pth")
    print(f"Done, saved to: {trainer.run_dir}", flush=True)


if __name__ == "__main__":
    sys.argv = _argparse_to_hydra_argv(sys.argv)
    main()
