"""The prior contract, the loader that streams batches off it, and the wrapping priors."""

from collections.abc import Callable, Iterator

import torch

from tfmplayground.utils import get_default_device

MAX_NUM_CLASSES = 10  # tabpfnv2 paper target generation subsection, natively limited to at most 10 classes

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Prior:
    """
    A prior samples synthetic tables and owns its own train/test split.

    Every prior returns (X_train, y_train, X_test, y_test) from batch(). It says which side it
    sits on through problem_type, and how many classes it can produce through max_num_classes,
    so pretrainTFM can pick a criterion without knowing anything else about it.
    """

    problem_type: str | None = None
    max_num_classes: int | None = None

    def batch(self, batch_size: int) -> Batch:
        raise NotImplementedError


class PriorDataLoader:
    """
    Endless stream of batches off a prior. The caller decides how many to take.

    Args:
        prior (Prior): the prior to sample from.
        batch_size (int): number of tables per batch.
    """

    def __init__(self, prior: Prior, batch_size: int):
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Batch]:
        while True:
            yield self.prior.batch(self.batch_size)


class FunctionPrior(Prior):
    """
    Prior backed by your own get_batch function.

    The function is called as get_batch_function(batch_size, num_datapoints_max, num_features)
    and must return (X_train, y_train, X_test, y_test). When problem is set it is passed on as
    a keyword argument, so functions that know nothing about problems keep working.

    Args:
        get_batch_function (Callable): a function returning batches of data.
        num_datapoints_max (int): max sequence length per table.
        num_features (int): number of input features.
        device (torch.device): device the batches end up on, defaults to the best available one.
        problem (str): "classification" or "regression", reported as problem_type.
        max_num_classes (int): highest number of classes the function can produce.
    """

    def __init__(
        self,
        get_batch_function: Callable[..., Batch],
        num_datapoints_max: int,
        num_features: int,
        device: torch.device = None,
        problem: str | None = None,
        max_num_classes: int | None = None,
    ):
        self.get_batch_function = get_batch_function
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device if device is not None else get_default_device()
        self.problem_type = problem
        self.max_num_classes = max_num_classes

    def batch(self, batch_size: int) -> Batch:
        problem = {} if self.problem_type is None else {"problem": self.problem_type}
        return self.get_batch_function(batch_size, self.num_datapoints_max, self.num_features, **problem)


class DictPrior(Prior):
    """
    Adapts a dataloader that yields dicts of x, y, target_y and train_test_split_index.

    The external prior wrappers speak that older shape, because the libraries behind them do.
    This keeps them usable without a rewrite. Only the halves the training loop reads survive
    the conversion: the train half of y and the test half of target_y.

    Args:
        loader: any iterable yielding the dict shape.
        problem (str): "classification" or "regression", read off the loader when not given.
        max_num_classes (int): highest number of classes, read off the loader when not given.
    """

    def __init__(self, loader, problem: str | None = None, max_num_classes: int | None = None):
        self.loader = loader
        self.batches = iter(loader)
        self.problem_type = problem if problem is not None else getattr(loader, "problem_type", None)
        self.max_num_classes = (
            max_num_classes if max_num_classes is not None else getattr(loader, "max_num_classes", None)
        )

    def batch(self, batch_size: int | None = None) -> Batch:
        try:
            d = next(self.batches)
        except StopIteration:
            self.batches = iter(self.loader)
            try:
                d = next(self.batches)
            except StopIteration as error:
                raise RuntimeError("the wrapped prior loader yielded no batches") from error
        sep = int(d["train_test_split_index"])
        x, y, target_y = d["x"], d["y"], d["target_y"]
        # the wrapped loader fixes its own batch size, so we only check the one we are given
        assert batch_size is None or x.shape[0] == batch_size, (
            f"the wrapped loader gave {x.shape[0]} tables, not {batch_size}"
        )
        return x[:, :sep], y[:, :sep], x[:, sep:], target_y[:, sep:]
