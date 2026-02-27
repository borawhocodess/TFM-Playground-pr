"""Hydra-based classification training entrypoint using BaseTrainer and PriorDumpDataset."""

import hydra
from omegaconf import DictConfig
from torch import nn

from sklearn.metrics import accuracy_score
from tfmplayground.training.callbacks import ConsoleLoggerCallback
from tfmplayground.training.trainer import BaseTrainer
from tfmplayground.training.util import tqdm_on_main
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_CLASSIFICATION
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dataset import PriorDumpDataset
from tfmplayground.utils import set_randomness_seed

set_randomness_seed(2402)


class ToyEvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks, device):
        self.tasks = tasks
        self.device = device

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, self.device)
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(accuracy_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        tqdm_on_main(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}"
        )


@hydra.main(version_base=None, config_path="configs", config_name="train_classification")
def main(cfg: DictConfig):
    dataset = PriorDumpDataset(
        **dict(cfg.dataset),
        num_steps=cfg.training.steps,
    )
    model = NanoTabPFNModel(
        **dict(cfg.model),
        num_outputs=dataset.max_num_classes,
    )
    device = str(cfg.dataset.device)
    callbacks = [ToyEvaluationLoggerCallback(TOY_TASKS_CLASSIFICATION, device)]
    trainer = BaseTrainer(
        model=model,
        train_dataset=dataset,
        criterion=nn.CrossEntropyLoss(),
        callbacks=callbacks,
        initial_lr=cfg.training.lr,
        accumulate_gradients=cfg.training.accumulate_gradients,
        epochs=cfg.training.epochs,
        steps=cfg.training.steps,
        run_name=cfg.training.run_name,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
    )
    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint", False))


if __name__ == "__main__":
    main()
