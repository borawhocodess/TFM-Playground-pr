"""Priors backed by an HDF5 dump, and the dumper that writes one."""

import h5py
import numpy as np
import torch
from tqdm import tqdm

from tfmplayground.priors.base import Batch, Prior
from tfmplayground.utils import get_default_device


class DumpPrior(Prior):
    """
    Prior that reads batches out of an HDF5 dump, and starts over when it runs out.

    Args:
        filename (str): path to the HDF5 file.
        device (torch.device): device the batches end up on, defaults to the best available one.
        starting_index (int): table to read first.
    """

    def __init__(self, filename, device: torch.device = None, starting_index: int = 0):
        self.filename = filename
        with h5py.File(self.filename, "r") as f:
            self.num_tables = f["X"].shape[0]
            self.num_datapoints_max = f["X"].shape[1]
            self.max_num_classes = f["max_num_classes"][0] if "max_num_classes" in f else None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
        self.device = device if device is not None else get_default_device()
        if not 0 <= starting_index < self.num_tables:
            raise ValueError(f"starting_index must be between 0 and {self.num_tables - 1}, got {starting_index}")
        self.pointer = starting_index

    def batch(self, batch_size: int) -> Batch:
        if not 0 < batch_size <= self.num_tables:
            raise ValueError(f"batch_size must be between 1 and {self.num_tables}, got {batch_size}")
        if self.pointer + batch_size > self.num_tables:
            self.pointer = 0

        with h5py.File(self.filename, "r") as f:
            end = self.pointer + batch_size
            num_features = f["num_features"][self.pointer : end].max()
            if self.has_num_datapoints:
                max_seq_in_batch = int(f["num_datapoints"][self.pointer : end].max())
            else:
                max_seq_in_batch = int(self.num_datapoints_max)

            x = torch.from_numpy(f["X"][self.pointer : end, :max_seq_in_batch, :num_features])
            y = torch.from_numpy(f["y"][self.pointer : end, :max_seq_in_batch])
            key = "train_test_split_index" if "train_test_split_index" in f else "single_eval_pos"
            sep = int(f[key][self.pointer : end][0])

            self.pointer += batch_size
            if self.pointer >= f["X"].shape[0]:
                print("Finished iteration over all stored datasets; reusing the same data.")
                self.pointer = 0

        x = x.to(self.device)
        y = y.to(self.device)
        return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


def dump_prior_to_h5(
    prior,
    num_batches: int,
    max_classes: int,
    batch_size: int,
    save_path: str,
    problem_type: str,
    max_seq_len: int,
    max_features: int,
):
    """Dumps synthetic prior data into an HDF5 file for later training."""

    with h5py.File(save_path, "w") as f:
        dump_X = f.create_dataset(
            "X",
            shape=(0, max_seq_len, max_features),
            maxshape=(None, max_seq_len, max_features),
            chunks=(batch_size, max_seq_len, max_features),
            compression="lzf",
        )
        dump_num_features = f.create_dataset(
            "num_features", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_num_datapoints = f.create_dataset(
            "num_datapoints", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_y = f.create_dataset(
            "y", shape=(0, max_seq_len), maxshape=(None, max_seq_len), chunks=(batch_size, max_seq_len)
        )
        dump_train_test_split_index = f.create_dataset(
            "train_test_split_index", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )

        if problem_type == "classification":
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for _ in tqdm(range(num_batches)):
            x_train, y_train, x_test, y_test = prior.batch(batch_size)
            train_test_split_index = x_train.shape[1]
            x = torch.cat([x_train, x_test], dim=1).to("cpu").numpy()
            y = torch.cat([y_train, y_test], dim=1).to("cpu").numpy()

            # pad x and y to the maximum sequence length and number of features needed for tabicl
            x_padded = np.pad(
                x, ((0, 0), (0, max_seq_len - x.shape[1]), (0, max_features - x.shape[2])), mode="constant"
            )
            y_padded = np.pad(y, ((0, 0), (0, max_seq_len - y.shape[1])), mode="constant")

            dump_X.resize(dump_X.shape[0] + batch_size, axis=0)
            dump_X[-batch_size:] = x_padded

            dump_y.resize(dump_y.shape[0] + batch_size, axis=0)
            dump_y[-batch_size:] = y_padded

            dump_num_features.resize(dump_num_features.shape[0] + batch_size, axis=0)
            dump_num_features[-batch_size:] = x.shape[2]

            dump_num_datapoints.resize(dump_num_datapoints.shape[0] + batch_size, axis=0)
            dump_num_datapoints[-batch_size:] = x.shape[1]

            dump_train_test_split_index.resize(dump_train_test_split_index.shape[0] + batch_size, axis=0)
            dump_train_test_split_index[-batch_size:] = train_test_split_index
