from typing import Any, Dict, Optional

import logging
import os
import time

from pfns.bar_distribution import FullSupportBarDistribution
import schedulefree
import torch
import torch.nn as nn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from .base import Callback, Trainer
from .util import (
    ddp_teardown,
    generate_run_id,
    infer_device,
    log_on_main,
)

logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):
    """Trainer class for training models with optional DDP support."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: IterableDataset,
        criterion: nn.Module,
        initial_lr: float = 1e-4,
        weight_decay: float = 0.0,
        accumulate_gradients: int = 1,
        epochs: int = 10000,
        steps: int = 100,
        callbacks: list[Callback] | None = None,
        run_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        task: Optional[str] = None,
        resume_dir: Optional[str] = None,
        use_cpu: bool = False,
        dataloader_num_workers: int = 0,
        **kwargs,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = callbacks if callbacks is not None else []

        self.run_dir = run_dir
        self.run_name = run_name
        self.task = task
        if resume_dir is not None:
            self.run_dir = resume_dir
        else:
            self._setup_output_dir()

        self.device, self.ddp = infer_device(use_cpu)

        if self.ddp:
            self._configure_ddp()
            self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.ddp_local_rank],
                broadcast_buffers=False,
            )
            self.raw_model = self.model.module
            # Batch size must be divisible by world size for DDP (no per-process batch split).
            if getattr(train_dataset, "batch_size", None) is not None:
                if train_dataset.batch_size % self.ddp_world_size != 0:
                    raise ValueError(
                        f"Dataset batch size {train_dataset.batch_size} not divisible by "
                        f"DDP world size {self.ddp_world_size}"
                    )
        else:
            self.master_process = True
            self.raw_model = self.model
            self.model.to(self.device)

        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.accumulate_gradients = accumulate_gradients
        self.epochs = epochs
        self.steps = steps
        self.criterion = criterion

        self.optimizer = schedulefree.AdamWScheduleFree(
            self.raw_model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )

        if dataloader_num_workers > 0:
            raise NotImplementedError(
                "dataloader_num_workers > 0 not supported yet (requires multiprocessing)."
            )

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=None,
            num_workers=dataloader_num_workers,
        )

    def _configure_ddp(self) -> None:
        init_process_group(backend="nccl")
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])
        self.master_process = self.ddp_rank == 0
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        log_on_main(
            logger,
            f"Running DDP with {self.ddp_world_size} processes",
            logging.INFO,
        )

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint from run_dir/latest_checkpoint.pth. Returns None if missing or invalid."""
        path = os.path.join(self.run_dir, "latest_checkpoint.pth")
        if not os.path.isfile(path):
            return None
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "epoch" not in checkpoint or "model" not in checkpoint:
            log_on_main(logger, f"Checkpoint at {path} missing required keys; starting from epoch 1", logging.WARNING)
            return None
        return checkpoint

    def _setup_output_dir(self) -> None:
        if self.run_dir is None:
            self.run_dir = "workdir/experiments"
        run_id = generate_run_id(run_name=self.run_name, task=self.task)
        if self.run_name is None:
            self.run_name = run_id
            self.run_dir = os.path.join(self.run_dir, self.run_name)
        else:
            # Named run: .../regression/<run_name>/<timestamp>-<task>-<uuid>-<run_name>/
            self.run_dir = os.path.join(self.run_dir, self.run_name, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    def _loss(self, output: torch.Tensor, targets: torch.Tensor) -> float:
        losses = self.criterion(output, targets)
        loss = losses.mean() / self.accumulate_gradients
        loss.backward()
        return loss.cpu().detach().item() * self.accumulate_gradients

    @ddp_teardown
    def train(self, resume_from_checkpoint: bool = False) -> nn.Module:
        checkpoint = None
        if resume_from_checkpoint:
            checkpoint = self._load_checkpoint()
            if checkpoint is not None:
                self.raw_model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

        classification_task = isinstance(self.criterion, nn.CrossEntropyLoss)
        regression_task = not classification_task

        assert self.steps % self.accumulate_gradients == 0, (
            "steps must be divisible by accumulate_gradients"
        )

        start_epoch = 1
        if checkpoint is not None:
            start_epoch = checkpoint["epoch"] + 1

        progress_bar = (
            tqdm(
                range(self.epochs),
                desc="Training",
                leave=True,
                initial=start_epoch - 1,
                total=self.epochs,
            )
            if self.master_process
            else range(self.epochs)
        )

        try:
            for epoch in range(start_epoch, self.epochs + 1):
                epoch_start_time = time.time()
                self.model.train()
                self.optimizer.train()
                total_loss = 0.0

                iterator = iter(self.train_dataloader)
                for step in range(self.steps):
                    full_data = next(iterator)
                    single_eval_pos = full_data["single_eval_pos"]
                    if isinstance(single_eval_pos, torch.Tensor):
                        single_eval_pos = single_eval_pos.item()
                    data = (
                        full_data["x"].to(self.device),
                        full_data["y"][:, :single_eval_pos].to(self.device),
                    )
                    if torch.isnan(data[0]).any() or torch.isnan(data[1]).any():
                        continue
                    targets = full_data["target_y"].to(self.device)

                    if regression_task:
                        y_mean = data[1].mean(dim=1, keepdim=True)
                        y_std = data[1].std(dim=1, keepdim=True) + 1e-8
                        y_norm = (data[1] - y_mean) / y_std
                        data = (data[0], y_norm)

                    output = self.model(data, single_eval_pos=single_eval_pos)
                    targets = targets[:, single_eval_pos:]
                    if regression_task:
                        targets = (targets - y_mean) / y_std
                    if classification_task:
                        targets = targets.reshape((-1,)).to(torch.long)
                        output = output.view(-1, output.shape[-1])

                    if self.ddp and (step + 1) % self.accumulate_gradients != 0:
                        with self.model.no_sync():
                            total_loss += self._loss(output, targets)
                    else:
                        total_loss += self._loss(output, targets)

                    if (step + 1) % self.accumulate_gradients == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.master_process:
                        progress_bar.set_postfix(
                            {
                                "epoch": f"{epoch}/{self.epochs}",
                                "step": f"{step + 1}/{self.steps}",
                            }
                        )

                end_time = time.time()
                mean_loss = total_loss / self.steps
                self.model.eval()
                self.optimizer.eval()

                training_state = {
                    "epoch": epoch,
                    "architecture": {
                        "num_layers": int(self.raw_model.num_layers),
                        "embedding_size": int(self.raw_model.embedding_size),
                        "num_attention_heads": int(self.raw_model.num_attention_heads),
                        "mlp_hidden_size": int(self.raw_model.mlp_hidden_size),
                        "num_outputs": int(self.raw_model.num_outputs),
                    },
                    "model": self.raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                torch.save(
                    training_state,
                    os.path.join(self.run_dir, "latest_checkpoint.pth"),
                )

                for callback in self.callbacks:
                    if type(self.criterion) is FullSupportBarDistribution:
                        callback.on_epoch_end(
                            epoch,
                            end_time - epoch_start_time,
                            mean_loss,
                            self.raw_model,
                            dist=self.criterion,
                        )
                    else:
                        callback.on_epoch_end(
                            epoch,
                            end_time - epoch_start_time,
                            mean_loss,
                            self.raw_model,
                        )

                if self.master_process:
                    progress_bar.update(1)

            if self.master_process:
                progress_bar.close()
        except KeyboardInterrupt:
            if self.master_process:
                progress_bar.close()
        finally:
            for callback in self.callbacks:
                callback.close()

        return self.raw_model
