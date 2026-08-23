import time
from collections.abc import Mapping

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground.callbacks import Callback, ConsoleLoggerCallback
from tfmplayground.models import NanoTabPFNModel, TabularFoundationModel
from tfmplayground.models.base import OUTPUT_KINDS
from tfmplayground.priors import Prior, PriorDataLoader
from tfmplayground.utils import FixedBinLoss, QuantileLoss, get_default_device, make_global_bucket_edges

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
            defaults to the official tabicl prior, sampled live, nothing to download
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
    # before inferring a criterion, because a model built for the other problem should say so
    # rather than fail on whatever its head happens to emit
    check_model_problem(model, problem)
    if criterion is None:
        criterion = infer_criterion(model, prior, device, problem)
    check_criterion_kind(model, criterion, problem)
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
    """
    The official tabicl prior, sampled on the fly, nothing to download.

    Held to small tables on purpose. tabicl's own defaults reach 1024 rows by 100 features, and
    a batch of those through the default model runs the mps allocator out of memory, so the no
    argument path would not even finish. Pass your own TabICLPrior for the full range.
    """
    from tfmplayground.external_priors import TabICLPrior

    return TabICLPrior(
        num_datapoints_min=129,
        num_datapoints_max=160,
        num_features_min=2,
        num_features_max=8,
        problem=problem,
        device=device,
    )


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
    # the shape of a regression head does not say what it means, so the model declares it. a
    # quantile head trained under a bar distribution has its outputs read as bucket logits, which
    # trains against the wrong objective without ever failing
    output_kind = getattr(model, "output_kind", "bar")
    if output_kind == "quantiles":
        return QuantileLoss(num_outputs)
    if output_kind == "fixed_bin_logits":
        borders = getattr(model, "regression_borders", None)
        if not callable(borders):
            raise ValueError(
                f"{type(model).__name__} declares fixed_bin_logits but has no regression_borders(), "
                "so there is nothing to say what its bins are"
            )
        # the model fixed these bins when it was built, so fitting new ones from the prior would
        # hand its channels ranges it never meant
        return FixedBinLoss(borders().to(device))
    if output_kind == "scalar":
        raise NotImplementedError(SCALAR_UNSUPPORTED.format(model=type(model).__name__))
    if output_kind != "bar":
        raise ValueError(f"output_kind must be one of {sorted(OUTPUT_KINDS)}, got {output_kind!r}")
    return FullSupportBarDistribution(make_global_bucket_edges(prior, n_buckets=num_outputs, device=device))


SCALAR_UNSUPPORTED = (
    "{model} declares a scalar regression head, one value per row, which upstream trains under "
    "mse. nothing here trains that yet, and passing your own criterion does not make it safe: a "
    "scalar head emits (batch, rows, 1) against targets of (batch, rows), which mse broadcasts "
    "into (batch, rows, rows) without complaint. build it with more than one output instead"
)

# what each head will accept. quantiles and fixed bins are structural, the criterion has to read
# the channels the way the model wrote them. "bar" is the tabpfn lineage's generic n logit head,
# where the meaning comes from the criterion, so more than one is legitimate there
CRITERION_KINDS = {
    "quantiles": (QuantileLoss,),
    "fixed_bin_logits": (FixedBinLoss,),
    "bar": (FullSupportBarDistribution, QuantileLoss),
}


def check_criterion_kind(model: TabularFoundationModel, criterion, problem: str | None):
    """
    The criterion has to match what the head emits, however the criterion got here.

    Inference reads output_kind, but a criterion handed in by the caller used to go unchecked, so
    a bar distribution with the right number of buckets went straight into a quantile head and
    trained against the wrong objective in silence.
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        return
    output_kind = getattr(model, "output_kind", "bar")
    if output_kind == "scalar":
        raise NotImplementedError(SCALAR_UNSUPPORTED.format(model=type(model).__name__))
    if output_kind == "class_logits":
        raise ValueError(
            f"{type(model).__name__} was built for classification, so its head holds class "
            f"logits, but {type(criterion).__name__} is a regression criterion"
        )
    expected = CRITERION_KINDS.get(output_kind)
    if expected is not None and not isinstance(criterion, expected):
        names = " or ".join(kind.__name__ for kind in expected)
        raise ValueError(
            f"{type(model).__name__} emits {output_kind}, which needs {names}, got {type(criterion).__name__}"
        )


def check_model_problem(model: TabularFoundationModel, problem: str | None):
    """
    A model is built for one problem or the other, and it says which, so ask rather than sniff.

    Feeding continuous targets to a class lookup does not reliably fail: cpu and cuda raise on the
    out of range indices, mps returns zeros and trains on a garbage y embedding.
    """
    if problem is None:
        return
    problems = getattr(model, "problems", PROBLEMS)
    if problem not in problems:
        raise ValueError(
            f"{type(model).__name__} was built for {' and '.join(problems)}, but the problem is "
            f"{problem!r}. build it for {problem} instead, which for the tabicl and tabfm family "
            "means max_classes=0 or is_classifier=False"
        )


def infer_num_outputs(model: TabularFoundationModel) -> int:
    """
    How many outputs the model produces per test row.

    Taken from the model when it declares one. The throwaway forward pass below is the fallback
    for third party models that do not, and it runs in whatever mode the model is in, which a
    custom model carrying running statistics would notice.
    """
    declared = getattr(model, "num_outputs", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    parameter = next(model.parameters())
    X_train = torch.randn(1, 8, 4, device=parameter.device, dtype=parameter.dtype)
    y_train = torch.randint(0, 2, (1, 8), device=parameter.device).to(parameter.dtype)
    X_test = torch.randn(1, 4, 4, device=parameter.device, dtype=parameter.dtype)
    # the model's own mode is left alone. tabicl branches on self.training at every level and
    # only builds the training path when the whole tree is in it, and no model here carries
    # running statistics, so a discarded no_grad forward changes nothing whichever mode it is in
    with torch.no_grad():
        out = model(X_train, y_train, X_test)
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
    status = "completed"

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
                    output = output.reshape(-1, output.shape[-1])  # tabicl hands back a non contiguous view

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

            # a callback that measures something returns it, and every callback after it is
            # handed what came before. that is how the eval score reaches a logger at all
            metrics = {}
            for callback in callbacks:
                reported = callback.on_epoch_end(
                    epoch,
                    end_time - epoch_start_time,
                    mean_loss,
                    (model.module if multi_gpu else model),
                    **{**metrics, "dist": criterion},  # dist is ours, a callback cannot shadow it
                )
                if isinstance(reported, Mapping):
                    metrics.update(reported)
                elif reported is not None:
                    raise TypeError(
                        f"{type(callback).__name__}.on_epoch_end returned "
                        f"{type(reported).__name__}, measurements have to be a mapping or nothing"
                    )
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = f"failed: {type(error).__name__}: {error}"
        raise
    finally:
        # a record that says a run finished when it crashed is worse than no record at all
        for callback in callbacks:
            callback.status = status
            callback.close()

    return (model.module if multi_gpu else model), mean_loss
