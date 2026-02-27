"""Data loading utilities for tabular priors."""

from typing import Callable, Dict, Iterator, Union

import torch
from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset
from ticl.dataloader import PriorDataLoader as TICLPriorDataset
# import here for future use & cleaner imports/it already handles type conversions
from tabpfn_prior import TabPFNPriorDataLoader
from torch.utils.data import DataLoader

from .dataset import PriorDumpDataset


class PriorDataLoader(DataLoader):
    """Generic DataLoader for synthetic data generation using a get_batch function.

    Args:
        get_batch_function (Callable): A function returning batches of data.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Max sequence length per function.
        num_features (int): Number of input features.
        device (torch.device): Device to move tensors to.
    """

    def __init__(
        self,
        get_batch_function: Callable[..., Dict[str, Union[torch.Tensor, int]]],
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device,
    ):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Union[torch.Tensor, int]]]:
        return iter(
            self.get_batch_function(self.batch_size, self.num_datapoints_max, self.num_features)
            for _ in range(self.num_steps)
        )

    def __len__(self) -> int:
        return self.num_steps


class PriorDumpDataLoader(DataLoader):
    """Thin wrapper around PriorDumpDataset for backward compatibility.

    Same __init__ and batch API as before; iteration is delegated to PriorDumpDataset.
    """

    def __init__(self, filename, num_steps, batch_size, device, starting_index=0):
        self._dataset = PriorDumpDataset(
            filename=filename,
            num_steps=num_steps,
            batch_size=batch_size,
            device=device,
            starting_index=starting_index,
        )
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.num_datapoints_max = self._dataset.num_datapoints_max
        self.max_num_classes = self._dataset.max_num_classes
        self.problem_type = self._dataset.problem_type

    def __iter__(self):
        return iter(self._dataset)

    def __len__(self):
        return self.num_steps


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
        prior_type (str): Type of prior: 'mlp_scm', 'tree_scm', 'mix_scm' (default), or 'dummy'.
        device (torch.device): Target device for tensors.
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
        device: torch.device,
        prior_type: str = "mix_scm",
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.prior_type = prior_type
        self.device = device

        self.pd = TabICLPriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
            prior_type=prior_type,
        )

    def tabicl_to_ours(self, d):
        x, y, active_features, seqlen, train_size = d
        active_features = active_features[
            0
        ].item()  # should be all the same since we use batch_size_per_gp=batch_size (not true in practice!)
        x = x[:, :, :active_features]
        single_eval_pos = train_size[0].item()  # should be all the same since we use batch_size_per_gp=batch_size
        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return iter(self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps


class TICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data from TICL's PriorDataLoader.

    Args:
        prior (Any): A TICL prior object supporting get_batch.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions sampled per batch.
        num_datapoints_max (int): Number of datapoints sampled per function.
        num_features (int): Dimensionality of x vectors.
        device (torch.device): Target device for tensors.
        min_eval_pos (int, optional): Minimum evaluation position in the sequence.
    """

    def __init__(
        self,
        prior,
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        min_eval_pos: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.device = device

        self.pd = TICLPriorDataset(
            prior=prior,
            num_steps=num_steps,
            batch_size=batch_size,
            min_eval_pos=min_eval_pos,
            n_samples=num_datapoints_max,
            device=device,
            num_features=num_features,
        )

    def ticl_to_ours(self, d):
        (info, x, y), target_y, single_eval_pos = d
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0)
        target_y = target_y.permute(1, 0)

        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=target_y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return (self.ticl_to_ours(batch) for batch in self.pd)

    def __len__(self):
        return self.num_steps
