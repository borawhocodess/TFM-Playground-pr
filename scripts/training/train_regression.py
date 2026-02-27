"""Hydra-based regression training entrypoint using BaseTrainer and PriorDumpDataset."""

import hydra
from omegaconf import DictConfig
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score
import torch

from tfmplayground.training.callbacks import ConsoleLoggerCallback
from tfmplayground.training.trainer import BaseTrainer
from tfmplayground.training.util import tqdm_on_main
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataset import PriorDumpDataset
from tfmplayground.utils import set_randomness_seed, make_global_bucket_edges

set_randomness_seed(2402)


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
    trainer = BaseTrainer(
        model=model,
        train_dataset=dataset,
        criterion=dist,
        callbacks=callbacks,
        initial_lr=cfg.training.lr,
        accumulate_gradients=cfg.training.accumulate_gradients,
        epochs=cfg.training.epochs,
        steps=cfg.training.steps,
        run_name=cfg.training.run_name,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
    )
    raw_model = trainer.train()

    # Save bucket edges and final model weights to run_dir
    torch.save(bucket_edges, f"{trainer.run_dir}/bucket_edges.pth")
    torch.save(raw_model.to("cpu").state_dict(), f"{trainer.run_dir}/final_model.pth")


if __name__ == "__main__":
    main()
