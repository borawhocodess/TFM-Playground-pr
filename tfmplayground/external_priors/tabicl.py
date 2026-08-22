"""The official TabICL prior, sampled live or dumped to HDF5."""

import torch
from tabicl.prior import PriorDataset as TabICLPriorDataset
from torch.utils.data import DataLoader

from tfmplayground.priors.base import MAX_NUM_CLASSES, Batch, Prior
from tfmplayground.utils import get_default_device

PROBLEMS = ("classification", "regression")


def build_dataset(
    problem: str,
    batch_size: int,
    num_datapoints_min: int,
    num_datapoints_max: int,
    num_features_min: int,
    num_features_max: int,
    max_num_classes: int,
    prior_type: str,
    n_jobs: int,
):
    """The library samples both problems off the same causal models, regression is one flag."""
    return TabICLPriorDataset(
        regression=(problem == "regression"),
        batch_size=batch_size,
        batch_size_per_gp=batch_size,
        min_features=num_features_min,
        max_features=num_features_max,
        max_classes=max_num_classes,
        min_seq_len=num_datapoints_min,
        max_seq_len=num_datapoints_max,
        prior_type=prior_type,
        n_jobs=n_jobs,
    )


def sample_table(dataset: TabICLPriorDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One draw off a tabicl dataset, narrowed to the features that carry signal."""
    x, y, active_features, _, train_size = next(dataset)  # endless, and it silences its own printing
    # every table in the batch shares these because we sample one group at a time (not true in practice!)
    x = x[:, :, : active_features[0].item()]
    return x.to(device), y.to(device), train_size[0].item()


class TabICLPrior(Prior):
    """
    The official tabicl prior, sampled live, no dump in between.

    Wraps the library's own sampler, which is endless, so this prior is too. It owns the
    train/test split and picks a new sequence length and feature count per batch, so tables
    change shape from one batch to the next. Both problems come off the same causal models, the
    library turns the target into classes only when regression is off.

    The dataset is built on the first batch and rebuilt if the batch size changes, because
    tabicl fixes that at construction and we only learn it when pretrainTFM asks for one.

    Args:
        num_datapoints_min (int): shortest table the library may draw.
        num_datapoints_max (int): longest table the library may draw.
        num_features_min (int): fewest features a table may carry.
        num_features_max (int): most features a table may carry.
        problem (str): "classification" or "regression".
        max_num_classes (int): most classes a classification target is cut into, at least 2.
        prior_type (str): "mlp_scm", "tree_scm", "mix_scm" or "dummy".
        n_jobs (int): worker processes the library samples with. 1, because the parallel path
            pickles its hyperparameter samplers and they are local closures, so anything above 1
            raises. tabicl says as much itself: the loky backend is for dumping to disk first.
        device (torch.device): device the batches end up on, defaults to the best available one.
    """

    def __init__(
        self,
        num_datapoints_min: int = 128,
        num_datapoints_max: int = 1024,
        num_features_min: int = 2,
        num_features_max: int = 100,
        problem: str = "classification",
        max_num_classes: int = MAX_NUM_CLASSES,
        prior_type: str = "mlp_scm",
        n_jobs: int = 1,
        device: torch.device = None,
    ):
        if num_datapoints_min >= num_datapoints_max:
            raise ValueError(
                f"num_datapoints_min must be smaller than num_datapoints_max, "
                f"got {num_datapoints_min} and {num_datapoints_max}"
            )
        if num_features_min > num_features_max:
            raise ValueError(
                f"num_features_min must not exceed num_features_max, got {num_features_min} and {num_features_max}"
            )
        if problem not in PROBLEMS:
            raise ValueError(f"problem must be one of {sorted(PROBLEMS)}, got {problem!r}")
        if problem == "classification" and max_num_classes < 2:
            raise ValueError(f"max_num_classes must be at least 2, got {max_num_classes}")
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.num_features_min = num_features_min
        self.num_features_max = num_features_max
        self.prior_type = prior_type
        self.n_jobs = n_jobs
        self.device = device if device is not None else get_default_device()
        self.problem_type = problem
        self.max_num_classes = max_num_classes if problem == "classification" else None
        self.dataset = None
        self.batch_size = None

    def batch(self, batch_size: int) -> Batch:
        # tabicl fixes the batch size when the dataset is built, and building one is free, so a
        # caller asking for a different size just gets a new dataset
        if batch_size != self.batch_size:
            self.dataset = build_dataset(
                problem=self.problem_type,
                batch_size=batch_size,
                num_datapoints_min=self.num_datapoints_min,
                num_datapoints_max=self.num_datapoints_max,
                num_features_min=self.num_features_min,
                num_features_max=self.num_features_max,
                max_num_classes=self.max_num_classes or 2,
                prior_type=self.prior_type,
                n_jobs=self.n_jobs,
            )
            self.batch_size = batch_size
        x, y, split = sample_table(self.dataset, self.device)
        return x[:, :split], y[:, :split], x[:, split:], y[:, split:]


class TabICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data on-the-fly from TabICL's PriorDataset.

    Args:
        num_steps (int): Number of batches to generate per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_min (int): Minimum number of datapoints per function.
        num_datapoints_max (int): Maximum number of datapoints per function.
        min_features (int): Minimum number of features in x.
        max_features (int): Maximum number of features in x.
        max_num_classes (int): Maximum number of classes (for classification tasks).
        problem (str): "classification" or "regression".
        prior_type (str): Type of prior: 'mlp_scm', 'tree_scm', 'mix_scm' (default), or 'dummy'.
        device (torch.device): Target device for tensors, defaults to the best available device.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        num_datapoints_min: int,
        num_datapoints_max: int,
        min_features: int,
        max_features: int,
        max_num_classes: int,
        device: torch.device = None,
        prior_type: str = "mix_scm",
        problem: str = "classification",
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.prior_type = prior_type
        self.device = device if device is not None else get_default_device()

        self.problem = problem
        self.pd = build_dataset(
            problem=problem,
            batch_size=batch_size,
            num_datapoints_min=num_datapoints_min,
            num_datapoints_max=num_datapoints_max,
            num_features_min=min_features,
            num_features_max=max_features,
            max_num_classes=max_num_classes or 2,
            prior_type=prior_type,
            n_jobs=-1,
        )

    def tabicl_to_ours(self):
        x, y, train_test_split_index = sample_table(self.pd, self.device)
        return dict(
            x=x,
            y=y,
            target_y=y,  # target_y is identical to y (for downstream compatibility)
            train_test_split_index=train_test_split_index,
        )

    def __iter__(self):
        return iter(self.tabicl_to_ours() for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps
