import h5py
import torch
from torch.utils.data import IterableDataset


class PriorDumpDataset(IterableDataset):
    """IterableDataset that loads synthetic prior data from an HDF5 dump.

    It mirrors the behavior of PriorDumpDataLoader but exposes the data as an
    iterable dataset, which can be wrapped by a DataLoader.
    """

    def __init__(self, filename, num_steps, batch_size, device, starting_index=0):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f["X"].shape[0]
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            self.stored_max_seq_len = f["X"].shape[1]
        self.device = device
        self.pointer = starting_index

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = f["num_features"][self.pointer:end].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x = torch.from_numpy(
                    f["X"][self.pointer:end, :max_seq_in_batch, :num_features]
                )
                y = torch.from_numpy(
                    f["y"][self.pointer:end, :max_seq_in_batch]
                )
                single_eval_pos = f["single_eval_pos"][self.pointer:end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print(
                        "Finished iteration over all stored datasets! "
                        "Will start reusing the same data with different splits now."
                    )
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),
                    single_eval_pos=single_eval_pos[0].item(),
                )

    def __len__(self):
        return self.num_steps

