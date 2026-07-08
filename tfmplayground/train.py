import time

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn
from torch.utils.data import DataLoader

from tfmplayground.callbacks import Callback, ConsoleLoggerCallback
from tfmplayground.models import TabularFoundationModel
from tfmplayground.utils import QuantileLoss, get_default_device, make_global_bucket_edges


def pretrainTFM(
    model: TabularFoundationModel,
    prior: DataLoader,
    eval: Callback | list[Callback] | None = None,
    regime=None,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss | None = None,
    epochs: int = 100,
    accumulate_gradients: int = 1,
    lr: float = 1e-4,
    device: torch.device = None,
    multi_gpu: bool = False,
) -> TabularFoundationModel:
    """
    Pretrains a tabular foundation model on a prior and hands back the trained model, nothing else to wire up.

    Args:
        model: (TabularFoundationModel) any model implementing the base forward contract
        prior: (DataLoader) torch-compatible dataloader providing the pretraining data
        eval: a callback or list of callbacks run at the end of each epoch,
            defaults to logging the loss to the console
        regime: reserved for training regimes, not implemented yet
        criterion: our loss criterion, inferred from the prior and model if not given
        epochs: (int) the number of epochs we train for
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        lr: (float) the learning rate
        device: (torch.device) the device we are using
        multi_gpu: (bool) whether to wrap the model in DataParallel

    Returns:
        (TabularFoundationModel) the trained model
    """
    if regime is not None:
        raise NotImplementedError("training regimes are not a thing yet")
    if device is None:
        device = get_default_device()
    if criterion is None:
        criterion = infer_criterion(model, prior, device)
    if eval is None:
        eval = [ConsoleLoggerCallback()]
    elif isinstance(eval, Callback):
        eval = [eval]
    trained_model, _ = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=epochs,
        accumulate_gradients=accumulate_gradients,
        lr=lr,
        device=device,
        callbacks=eval,
        multi_gpu=multi_gpu,
    )
    return trained_model


def infer_criterion(
    model: TabularFoundationModel, prior: DataLoader, device: torch.device
) -> nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss:
    """
    Picks a loss criterion based on what the prior provides: cross entropy for classification,
    a bar distribution fitted on the dump for regression dumps and a quantile loss otherwise.
    The number of model outputs is probed with a tiny forward pass.
    """
    problem_type = getattr(prior, "problem_type", None)
    if problem_type == "classification" or (problem_type is None and getattr(prior, "max_num_classes", None)):
        return nn.CrossEntropyLoss()
    num_outputs = infer_num_outputs(model)
    filename = getattr(prior, "filename", None)
    if filename is not None:
        bucket_edges = make_global_bucket_edges(filename, n_buckets=num_outputs, device=device)
        return FullSupportBarDistribution(bucket_edges)
    return QuantileLoss(num_outputs)


def infer_num_outputs(model: TabularFoundationModel) -> int:
    """Runs a tiny forward pass to find out how many outputs the model produces per test row."""
    parameter = next(model.parameters())
    X_train = torch.randn(1, 8, 4, device=parameter.device, dtype=parameter.dtype)
    y_train = torch.randn(1, 8, device=parameter.device, dtype=parameter.dtype)
    X_test = torch.randn(1, 4, 4, device=parameter.device, dtype=parameter.dtype)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(X_train, y_train, X_test)
    model.train(was_training)
    return out.shape[-1]


def train(
    model: TabularFoundationModel,
    prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss,
    epochs: int,
    accumulate_gradients: int = 1,
    lr: float = 1e-4,
    device: torch.device = None,
    callbacks: list[Callback] = None,
    multi_gpu: bool = False,
):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (TabularFoundationModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for,
            the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task

    assert prior.num_steps % accumulate_gradients == 0, "num_steps must be divisible by accumulate_gradients"

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            for i, full_data in enumerate(prior):
                train_test_split_index = full_data["train_test_split_index"]
                x = full_data["x"].to(device)
                y_train = full_data["y"][:, :train_test_split_index].to(device)
                if torch.isnan(x).any() or torch.isnan(y_train).any():
                    continue
                targets = full_data["target_y"].to(device)

                if regression_task:
                    y_mean = y_train.mean(dim=1, keepdim=True)
                    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
                    y_train = (y_train - y_mean) / y_std

                output = model(x[:, :train_test_split_index], y_train, x[:, train_test_split_index:])
                targets = targets[:, train_test_split_index:]
                if regression_task:
                    targets = (targets - y_mean) / y_std
                if classification_task:
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])

                losses = criterion(output, targets)
                loss = losses.mean() / accumulate_gradients
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            model.eval()
            optimizer.eval()

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        (model.module if multi_gpu else model),
                        dist=criterion,
                    )
                else:
                    callback.on_epoch_end(
                        epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model)
                    )
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), total_loss
