from abc import ABC, abstractmethod
from datetime import datetime

from sklearn.metrics import r2_score, roc_auc_score

from tfmplayground.evaluation.evaluation import get_openml_predictions
from tfmplayground.interface import TabularClassifier, TabularRegressor


class Callback(ABC):
    """Abstract base class for callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            epoch_time (float): Time of the epoch in seconds.
            loss (float): Mean loss for the epoch.
            model: The model being trained.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Called to release any resources or perform cleanup.
        """
        pass


class BaseLoggerCallback(Callback):
    """Abstract base class for logger callbacks."""

    pass


class ConsoleLoggerCallback(BaseLoggerCallback):
    """Logger callback that prints epoch information to the console."""

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        print(f"Epoch {epoch:5d} | Time {epoch_time:5.2f}s | Mean Loss {loss:5.2f}", flush=True)

    def close(self):
        """Nothing to clean up for print logger."""
        pass


class TensorboardLoggerCallback(BaseLoggerCallback):
    """Logger callback that logs epoch information to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Time/epoch", epoch_time, epoch)

    def close(self):
        self.writer.close()


class WandbLoggerCallback(BaseLoggerCallback):
    """Logger callback that logs epoch information to Weights & Biases."""

    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        """
        Initializes a WandbLoggerCallback.

        Args:
            project (str): The name of the wandb project.
            name (str, optional): The name of the run. Defaults to None.
            config (dict, optional): Configuration dictionary for the run. Defaults to None.
            log_dir (str, optional): Directory to save wandb logs. Defaults to None.
        """
        try:
            import wandb

            self.wandb = wandb  # store wandb module to avoid import if not used
            wandb.init(project=project, name=name, id=name, config=config, dir=log_dir, resume="allow")
        except ImportError as e:
            raise ImportError("wandb is not installed. Install it with: pip install wandb") from e

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        log_dict = {"epoch": epoch, "loss": loss, " epoch_time": epoch_time}
        self.wandb.log(log_dict)

    def close(self):
        self.wandb.finish()


class ExperimentCallback(BaseLoggerCallback):
    def __init__(self, experiment):
        self.experiment = experiment
        self.experiment.print0(f"experiment: {self.experiment.id}", console=True)

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.experiment.print0(f"e:{epoch} l:{loss:.4f} e_t:{epoch_time:.2f}s")

    def close(self):
        minutes = (datetime.now() - self.experiment.started).total_seconds() / 60
        self.experiment.print0(f"runtime: {minutes:.2f} mins")


class ExperimentEvaluationCallback(ExperimentCallback):
    def __init__(self, experiment, config, tasks, device):
        super().__init__(experiment)
        self.config = config
        self.tasks = tasks
        self.device = device

    def predictions(self, model):
        return get_openml_predictions(
            model=model,
            tasks=self.tasks,
            max_n_features=self.config.max_n_features,
            max_n_samples=self.config.max_n_samples,
        )

    def evaluate(self, model, **kwargs):
        raise NotImplementedError

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        scores = self.evaluate(model, **kwargs)
        if not scores:
            raise ValueError("none of the tasks fits this model, so there is nothing to score")
        mean = sum(scores) / len(scores)
        line = f"e:{epoch} l:{loss:.4f} e_t:{epoch_time:.2f}s {self.metric}:{mean:.4f} t:{len(scores)}"
        self.experiment.print0(line, console=True)


class ClassifierExperimentEvaluationCallback(ExperimentEvaluationCallback):
    metric = "roc_auc"

    def evaluate(self, model, **kwargs):
        classifier = TabularClassifier(model, self.device)
        predictions = self.predictions(classifier)
        scores = [roc_auc_score(y_true, y_proba, multi_class="ovr") for y_true, _, y_proba in predictions.values()]
        return scores


class RegressorExperimentEvaluationCallback(ExperimentEvaluationCallback):
    metric = "r2"

    def evaluate(self, model, **kwargs):
        regressor = TabularRegressor(model, kwargs.get("dist"), self.device)
        predictions = self.predictions(regressor)
        scores = [r2_score(y_true, y_pred) for y_true, y_pred, _ in predictions.values()]
        return scores
