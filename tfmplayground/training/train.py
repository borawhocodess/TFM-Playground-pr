import time

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground.models import TabularFoundationModel
from tfmplayground.priors import Prior, PriorDataLoader
from tfmplayground.training.callbacks import Callback
from tfmplayground.utils import QuantileLoss, get_default_device


def train(
    model: TabularFoundationModel,
    prior: Prior,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    lr: float,
    grad_clip: float,
    device: torch.device = None,
    callbacks: list[Callback] = None,
):
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    criterion = criterion.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task
    batches = iter(PriorDataLoader(prior, batch_size))

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            num_valid = 0
            for _ in range(steps_per_epoch):
                x_train, y_train, x_test, y_test = next(batches)
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                if not all(torch.isfinite(t).all() for t in (x_train, y_train, x_test, y_test)):
                    continue

                if regression_task:
                    y_mean = y_train.mean(dim=1, keepdim=True)
                    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
                    y_train = (y_train - y_mean) / y_std
                    y_test = (y_test - y_mean) / y_std
                    if not torch.isfinite(y_train).all() or not torch.isfinite(y_test).all():
                        continue

                output = model(x_train, y_train, x_test)
                if not torch.isfinite(output).all():
                    continue
                if classification_task:
                    y_test = y_test.reshape((-1,)).to(torch.long)
                    output = output.reshape(-1, output.shape[-1])
                    outputs = output.shape[-1]
                    biggest = int(y_test.max())
                    if biggest >= outputs:
                        raise ValueError(f"the prior makes class {biggest} but the model has {outputs} outputs")

                losses = criterion(output, y_test)
                if not torch.isfinite(losses).all():
                    continue

                loss = losses.mean()
                loss.backward()
                total_loss += loss.cpu().detach().item()
                num_valid += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if num_valid == 0:
                raise RuntimeError("the prior gave no finite batches in this epoch")

            end_time = time.time()
            mean_loss = total_loss / num_valid
            model.eval()
            optimizer.eval()

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        model,
                        dist=criterion,
                    )
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model)
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return model, mean_loss
