import h5py
import numpy as np
import pytest
import torch

from tfmplayground.priors import DumpPrior, Prior


def write_dump(path, num_groups=3, group_size=4, num_rows=10, num_features=3, splits=(3, 6, 4)):
    num_tables = num_groups * group_size
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=rng.standard_normal((num_tables, num_rows, num_features), dtype=np.float32))
        f.create_dataset("y", data=rng.integers(0, 3, size=(num_tables, num_rows)).astype(np.float32))
        f.create_dataset("num_features", data=np.full(num_tables, num_features, dtype="i4"))
        f.create_dataset("train_test_split_index", data=np.repeat(splits, group_size).astype("i4"))
        f.create_dataset("original_batch_size", data=np.array([group_size]))
    return path


@pytest.fixture
def dump(tmp_path):
    return write_dump(tmp_path / "dump.h5")


def test_batch_gives_four_tuples_split_where_the_dump_says(dump):
    prior = DumpPrior(filename=dump, device=torch.device("cpu"))

    assert isinstance(prior, Prior)
    x_train, y_train, x_test, y_test = prior.batch(4)

    assert x_train.shape == (4, 3, 3)
    assert y_train.shape == (4, 3)
    assert x_test.shape == (4, 7, 3)
    assert y_test.shape == (4, 7)


def test_pointer_wraps_at_the_end_of_the_dump(dump):
    prior = DumpPrior(filename=dump, device=torch.device("cpu"))

    for _ in range(3):
        prior.batch(4)
    assert prior.pointer == 12

    prior.batch(4)
    assert prior.pointer == 4


def test_a_batch_spanning_two_splits_is_refused(dump):
    prior = DumpPrior(filename=dump, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="spans tables split at"):
        prior.batch(6)


def test_starting_index_picks_the_first_table(dump):
    prior = DumpPrior(filename=dump, device=torch.device("cpu"), starting_index=4)

    x_train, _, _, _ = prior.batch(4)

    assert x_train.shape[1] == 6


def test_nanotabicl_varies_its_split_the_way_tabicl_does():
    from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig
    from tfmplayground.priors import NanoTabICLPrior

    prior = NanoTabICLPrior(config=NanoTabICLClassificationPriorConfig(num_datapoints_max=200), device="cpu")
    fractions = set()
    for _ in range(12):
        _, y_train, _, y_test = prior.batch(1)
        rows = y_train.shape[1] + y_test.shape[1]
        assert rows == 200
        fractions.add(y_train.shape[1] / rows)
    assert len(fractions) > 1
    assert all(0.1 <= fraction <= 0.9 for fraction in fractions)


def test_the_train_fractions_are_checked():
    from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig
    from tfmplayground.priors import NanoTabICLPrior

    for bad in (
        {"train_fraction_min": 0.0},
        {"train_fraction_max": 1.0},
        {"train_fraction_min": 0.8, "train_fraction_max": 0.2},
    ):
        with pytest.raises(ValueError, match="train fractions"):
            NanoTabICLPrior(config=NanoTabICLClassificationPriorConfig(**bad), device="cpu")
