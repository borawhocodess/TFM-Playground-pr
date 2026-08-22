import time

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground.callbacks import Callback, ConsoleLoggerCallback
from tfmplayground.models import NanoTabPFNModel, TabularFoundationModel
from tfmplayground.prior import Prior, PriorDataLoader
from tfmplayground.utils import QuantileLoss, get_default_device, make_global_bucket_edges

PROBLEMS = ("classification", "regression")

DUMP_URLS = {
    "classification": "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5",
    "regression": "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5",
}


def pretrainTFM(
    model: TabularFoundationModel | None = None,
    prior: Prior | None = None,
    eval: Callback | list[Callback] | None = None,
    regime=None,
    problem: str | None = None,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss | None = None,
    epochs: int = 100,
    steps_per_epoch: int = 25,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: torch.device = None,
    multi_gpu: bool = False,
) -> TabularFoundationModel:
    """
    Pretrains a tabular foundation model on a prior and hands back the trained model, nothing else to wire up.

    Args:
        model: (TabularFoundationModel) any model implementing the base forward contract,
            defaults to a nanotabpfn with a 10 class head (100 buckets for regression)
        prior: (Prior) the prior to pretrain on, sampled batch by batch,
            defaults to our own structural causal model prior, nothing to download
        eval: a callback or list of callbacks run at the end of each epoch,
            defaults to logging the loss to the console
        regime: reserved for training regimes, not implemented yet
        problem: "classification" or "regression", steers the default prior and criterion,
            inferred from the prior if not given
        criterion: our loss criterion, inferred from the prior and model if not given,
            must sit on the same side as the problem
        epochs: (int) the number of epochs we train for
        steps_per_epoch: (int) the number of batches that make up an epoch
        batch_size: (int) the number of tables per batch
        lr: (float) the learning rate
        device: (torch.device) the device we are using
        multi_gpu: (bool) whether to wrap the model in DataParallel

    Returns:
        (TabularFoundationModel) the trained model, carrying the fitted bar distribution
            as model.dist when trained for regression
    """
    if regime is not None:
        raise NotImplementedError("training regimes are not a thing yet")
    if problem is not None and problem not in PROBLEMS:
        raise ValueError(f"problem must be one of {sorted(PROBLEMS)}, got {problem!r}")
    prior_problem = getattr(prior, "problem_type", None)
    if problem is not None and prior_problem is not None and problem != prior_problem:
        raise ValueError(f"problem={problem!r} but the prior says {prior_problem!r}")
    problem = problem or prior_problem
    if criterion is not None and problem is not None:
        criterion_problem = "classification" if isinstance(criterion, nn.CrossEntropyLoss) else "regression"
        if problem != criterion_problem:
            raise ValueError(
                f"criterion {type(criterion).__name__} is for {criterion_problem} but the problem is {problem!r}"
            )
    if device is None:
        device = get_default_device()
    if prior is None:
        prior = default_prior(device, problem or "classification")
    if problem is None:
        problem = getattr(prior, "problem_type", None)
    if model is None:
        model = default_model(problem)
    if criterion is None:
        criterion = infer_criterion(model, prior, device, problem)
    num_outputs = infer_num_outputs(model)
    max_num_classes = getattr(prior, "max_num_classes", None)
    if isinstance(criterion, nn.CrossEntropyLoss) and max_num_classes and int(max_num_classes) > num_outputs:
        raise ValueError(
            f"the prior holds up to {int(max_num_classes)} classes but the model has {num_outputs} outputs"
        )
    if isinstance(criterion, FullSupportBarDistribution) and criterion.borders.numel() - 1 != num_outputs:
        raise ValueError(
            f"the criterion has {criterion.borders.numel() - 1} buckets but the model has {num_outputs} outputs"
        )
    if isinstance(criterion, QuantileLoss) and criterion.alphas.numel() != num_outputs:
        raise ValueError(
            f"the criterion has {criterion.alphas.numel()} quantiles but the model has {num_outputs} outputs"
        )
    if eval is None:
        eval = [ConsoleLoggerCallback()]
    elif isinstance(eval, Callback):
        eval = [eval]
    for callback in eval:
        side = getattr(callback, "classification", None)
        if side is not None and problem is not None:
            side = "classification" if side else "regression"
            if side != problem:
                raise ValueError(
                    f"eval callback {type(callback).__name__} is set up for {side} but the problem is {problem!r}"
                )
    trained_model, _ = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        lr=lr,
        device=device,
        callbacks=eval,
        multi_gpu=multi_gpu,
    )
    decoder = getattr(criterion, "mean", None)
    if callable(decoder):
        trained_model.dist = criterion
    return trained_model


def default_prior(device: torch.device, problem: str) -> Prior:
    """Our own structural causal model prior, sampled on the fly, nothing to download."""
    from tfmplayground.prior import SCMPrior

    return SCMPrior(num_datapoints_max=160, num_features=8, problem=problem, device=device)


def default_model(problem: str | None) -> TabularFoundationModel:
    """Builds a nanotabpfn with a fixed 10 class head for classification and 100 buckets otherwise."""
    return NanoTabPFNModel(
        num_attention_heads=6,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=10 if problem == "classification" else 100,
    )


def infer_criterion(
    model: TabularFoundationModel, prior: Prior, device: torch.device, problem: str | None = None
) -> nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss:
    """
    Picks a loss criterion based on the problem or what the prior provides: cross entropy for classification,
    a bar distribution fitted on the targets of a regression prior and a quantile loss otherwise.
    The number of model outputs is probed with a tiny forward pass.
    """
    problem = problem or getattr(prior, "problem_type", None)
    if problem == "classification" or (problem is None and getattr(prior, "max_num_classes", None)):
        return nn.CrossEntropyLoss()
    num_outputs = infer_num_outputs(model)
    if problem == "regression" or getattr(prior, "filename", None) is not None:
        return FullSupportBarDistribution(make_global_bucket_edges(prior, n_buckets=num_outputs, device=device))
    return QuantileLoss(num_outputs)


def infer_num_outputs(model: TabularFoundationModel) -> int:
    """Runs a tiny forward pass to find out how many outputs the model produces per test row."""
    parameter = next(model.parameters())
    X_train = torch.randn(1, 8, 4, device=parameter.device, dtype=parameter.dtype)
    y_train = torch.randint(0, 2, (1, 8), device=parameter.device).to(parameter.dtype)
    X_test = torch.randn(1, 4, 4, device=parameter.device, dtype=parameter.dtype)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(X_train, y_train, X_test)
    model.train(was_training)
    return out.shape[-1]


def train(
    model: TabularFoundationModel,
    prior: Prior,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution | QuantileLoss,
    epochs: int,
    steps_per_epoch: int = 25,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: torch.device = None,
    callbacks: list[Callback] = None,
    multi_gpu: bool = False,
):
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

    batches = iter(PriorDataLoader(prior, batch_size))

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            num_valid = 0
            for _ in range(steps_per_epoch):
                x_train, y_train, x_test, targets = next(batches)
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                x_test = x_test.to(device)
                targets = targets.to(device)
                if not all(torch.isfinite(tensor).all() for tensor in (x_train, y_train, x_test, targets)):
                    continue

                if regression_task:
                    y_mean = y_train.mean(dim=1, keepdim=True)
                    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
                    y_train = (y_train - y_mean) / y_std
                    targets = (targets - y_mean) / y_std
                    # a single training row makes the unbiased std nan, which finite inputs do not reveal
                    if not torch.isfinite(y_train).all() or not torch.isfinite(targets).all():
                        continue

                output = model(x_train, y_train, x_test)
                # a model's own feature normalization can go non-finite on a one row context
                if not torch.isfinite(output).all():
                    continue

                if classification_task:
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])

                losses = criterion(output, targets)
                if not torch.isfinite(losses).all():
                    continue

                num_valid += 1
                loss = losses.mean()
                loss.backward()
                total_loss += loss.cpu().detach().item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if num_valid == 0:
                raise RuntimeError("the prior produced no finite batches in this epoch")

            end_time = time.time()
            mean_loss = total_loss / num_valid
            model.eval()
            optimizer.eval()

            for callback in callbacks:
                callback.on_epoch_end(
                    epoch,
                    end_time - epoch_start_time,
                    mean_loss,
                    (model.module if multi_gpu else model),
                    dist=criterion,
                )
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), mean_loss
