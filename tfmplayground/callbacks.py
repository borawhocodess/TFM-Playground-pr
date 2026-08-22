import os
import platform
import socket
import subprocess
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime

import torch

PROBLEMS = ("classification", "regression")


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
            **kwargs: the criterion as dist, plus whatever earlier callbacks measured.

        Returns:
            (dict | None) anything this callback measured, which every callback after it in the
            list is then handed. Return nothing if it only reports. Order matters because of
            this: put the callbacks that measure before the ones that write things down.
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

    @staticmethod
    def measurements(kwargs: dict) -> dict:
        """Whatever the callbacks before this one measured, without the plumbing they came with."""
        return {name: value for name, value in kwargs.items() if name != "dist"}


class ConsoleLoggerCallback(BaseLoggerCallback):
    """Logger callback that prints epoch information to the console."""

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        line = f"Epoch {epoch:5d} | Time {epoch_time:5.2f}s | Mean Loss {loss:5.2f}"
        for name, value in self.measurements(kwargs).items():
            line += f" | {name} {value:.3f}" if isinstance(value, float) else f" | {name} {value}"
        print(line, flush=True)

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
        for name, value in self.measurements(kwargs).items():
            if isinstance(value, int | float):
                self.writer.add_scalar(f"Eval/{name}", value, epoch)

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
        self.wandb.log({"epoch": epoch, "loss": loss, "epoch_time": epoch_time, **self.measurements(kwargs)})

    def close(self):
        self.wandb.finish()


class ExperimentCallback(BaseLoggerCallback):
    """
    One directory per run, holding everything needed to say what that run was.

    Follows the modded nanotabpfn scripts:
    workdir/experiments/<problem>/<name>/<e_id>/<e_id>-log.txt, where e_id is a timestamp, a short
    uid and the name, so runs sort by time and never collide. Classification and regression runs
    live apart because they are not comparable, different criteria and different metrics.
    The log opens with the environment and the config, carries a line per epoch, and closes with
    the runtime, so the file answers "what was this run" without reference to anything else.

    Checkpointing is deliberately not here yet, see workdir/notes/reimplement.md.

    Args:
        name (str): names the run and the directory it goes in. Empty is allowed.
        problem (str): "classification" or "regression", the folder runs are grouped under.
        experiments_dir (str): where run directories are created.
        config: a dataclass or dict of every knob the run was given, dumped into the log.
        console (bool): whether the per-epoch line also goes to stdout.
        source (str | None): a file to copy into the log, defaults to the script that started
            this. The modded scripts write their own source into the log so a run can be read
            back years later without guessing which version of the code produced it.
    """

    def __init__(
        self,
        name: str = "",
        problem: str = "classification",
        experiments_dir: str = "workdir/experiments",
        config=None,
        console: bool = True,
        source: str | None = None,
    ):
        if problem not in PROBLEMS:
            raise ValueError(f"problem must be one of {sorted(PROBLEMS)}, got {problem!r}")
        self.started = datetime.now()
        self.console = console
        self.problem = problem
        timestamp = self.started.strftime("%y%m%d-%H%M%S")
        uid = uuid.uuid4().hex[:8]
        name = name.strip()
        self.e_id = f"{timestamp}-{uid}-{name}" if name else f"{timestamp}-{uid}"
        root = os.path.join(experiments_dir, problem)
        root = os.path.join(root, name) if name else root
        self.e_dir = os.path.join(root, self.e_id)
        os.makedirs(self.e_dir, exist_ok=True)
        self.log_path = os.path.join(self.e_dir, f"{self.e_id}-log.txt")
        self.epochs = 0

        source = source if source is not None else sys.argv[0]
        if source and os.path.isfile(source):
            try:  # sys.argv[0] is whatever launched us, which is not always readable text
                with open(source) as handle:
                    self.print0(handle.read())
                self.print0("=" * 100)
            except (OSError, UnicodeDecodeError):
                pass
        self.print0(f"experiment: {self.e_id}", console=True)
        self.print0(f"problem: {problem}")
        self.print0(f"start timestamp: {self.started.strftime('%Y-%m-%d %H:%M:%S')}")
        self.print0(f"host: {socket.gethostname()}")
        self.print0(f"platform: {platform.platform()}")
        self.print0(f"python: {sys.version}")
        self.print0(f"torch: {torch.__version__}")
        self.print0(f"cuda: {torch.version.cuda}")
        self.print0(f"mps: {torch.backends.mps.is_available()}")
        if torch.cuda.is_available():
            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            self.print0(smi.stdout)
        if config is not None:
            self.print0("=" * 100)
            self.print0("config:")
            settings = asdict(config) if is_dataclass(config) else dict(config)
            for setting, value in settings.items():
                self.print0(f"  {setting}: {value}")
        self.print0("=" * 100)

    def print0(self, line: str, console: bool = False):
        """Always to the log file, to the console only when asked."""
        with open(self.log_path, "a") as handle:
            print(line, file=handle)
        if console:
            print(line, flush=True)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        if self.epochs == 0:
            total = sum(parameter.numel() for parameter in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print0(f"model: {type(model).__name__}")
            self.print0(f"params: {total:,} (trainable: {trainable:,})")
            self.print0("=" * 100)
        self.epochs = epoch

        line = f"e:{epoch} l:{loss:.4f} e_t:{epoch_time:.2f}s"
        for name, value in self.measurements(kwargs).items():
            line += f" {name}:{value:.4f}" if isinstance(value, float) else f" {name}:{value}"
        self.print0(line, console=self.console)

    def close(self):
        ended = datetime.now()
        self.print0("=" * 100)
        if torch.cuda.is_available():
            self.print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
            self.print0(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
        self.print0(f"epochs: {self.epochs}")
        self.print0(f"end timestamp: {ended.strftime('%Y-%m-%d %H:%M:%S')}")
        self.print0(f"runtime: {(ended - self.started).total_seconds() / 60:.2f} mins")
        self.print0(f"experiment done: {self.e_id}", console=True)
