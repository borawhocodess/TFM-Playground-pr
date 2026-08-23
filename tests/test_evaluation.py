import numpy as np
import pandas as pd
import pytest

from tfmplayground.evaluation import (
    OpenMLEvaluationCallback,
    can_stratify,
    cross_validation_splits,
    get_openml_predictions,
    shrink_task,
)


def make_task(rows=5000, columns=300, classification=True, weights=(0.8, 0.15, 0.05)):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(rows, columns)))
    if classification:
        y = pd.Series(rng.choice(len(weights), size=rows, p=weights))
    else:
        y = pd.Series(rng.normal(size=rows))
    return X, y


def test_shrink_task_cuts_both_dimensions():
    """An oversized task is cut down to the limits rather than dropped."""
    X, y = make_task()
    X_small, y_small = shrink_task(X, y, max_n_features=100, max_n_samples=1000, classification=True, seed=11)

    assert X_small.shape == (1000, 100)
    assert len(y_small) == 1000
    assert list(X_small.index) == list(range(1000))  # reset, so fold indices line up


def test_shrink_task_keeps_the_class_balance():
    """An unstratified sample of an imbalanced task can lose a class, and roc auc then raises."""
    X, y = make_task()
    _, y_small = shrink_task(X, y, max_n_features=100, max_n_samples=1000, classification=True, seed=11)

    before = y.value_counts(normalize=True).sort_index().to_numpy()
    after = y_small.value_counts(normalize=True).sort_index().to_numpy()
    assert set(y_small.unique()) == set(y.unique())
    assert np.allclose(before, after, atol=0.01)


def test_shrink_task_leaves_a_small_task_alone():
    """Under the limits nothing is touched, so skip and subsample agree on small tasks."""
    X, y = make_task(rows=50, columns=4)
    X_small, y_small = shrink_task(X, y, max_n_features=100, max_n_samples=1000, classification=True, seed=11)

    assert X_small.shape == (50, 4)
    assert y_small.equals(y)


def test_shrink_task_falls_back_when_a_class_is_too_rare():
    """A class with a single member cannot be stratified, and that must not raise."""
    X, _ = make_task()
    y = pd.Series([0] * 4999 + [1])
    assert not can_stratify(y, 2)

    X_small, y_small = shrink_task(X, y, max_n_features=100, max_n_samples=1000, classification=True, seed=11)
    assert X_small.shape == (1000, 100)


def test_shrink_task_is_seeded():
    """The score has to move because the model moved, not because the sample did."""
    X, y = make_task()
    first = shrink_task(X, y, 100, 1000, True, seed=11)[0]
    same = shrink_task(X, y, 100, 1000, True, seed=11)[0]
    other = shrink_task(X, y, 100, 1000, True, seed=12)[0]

    assert first.equals(same)
    assert not first.equals(other)


@pytest.mark.parametrize("classification", [True, False])
def test_cross_validation_splits_test_every_row_once(classification):
    """Folds are what buys the signal back from a small sample, so they have to cover it."""
    X, y = make_task(rows=200, columns=4, classification=classification, weights=(0.5, 0.3, 0.2))
    splits = cross_validation_splits(X, y, folds=5, classification=classification, seed=11)

    assert len(splits) == 5
    tested = np.concatenate([test for _, test in splits])
    assert sorted(tested) == list(range(len(X)))
    for train, test in splits:
        assert not set(train) & set(test)


def test_a_task_too_rare_to_stratify_is_skipped_rather_than_folded_blindly():
    """
    Plain KFold splits without looking at y, so a class too rare to stratify can sit entirely in
    one test fold. The label encoder is fitted on the training half and then meets a label it has
    never seen. Measured over 200 seeds at 300 rows and five folds, a one row class does this
    every time. The eval callback runs every epoch, so that kills the run, not just the score.
    """
    X, _ = make_task(rows=200, columns=4)
    y = pd.Series([0] * 197 + [1, 1, 1])
    assert not can_stratify(y, 5)

    assert cross_validation_splits(X, y, folds=5, classification=True, seed=11) == []


def test_every_returned_fold_has_its_classes_in_both_halves():
    """The point of skipping: whatever comes back must be safe to fit an encoder on."""
    X, _ = make_task(rows=200, columns=4)
    y = pd.Series([0] * 100 + [1] * 90 + [2] * 10)

    splits = cross_validation_splits(X, y, folds=5, classification=True, seed=11)
    assert len(splits) == 5
    for train, test in splits:
        assert not set(y.iloc[test]) - set(y.iloc[train])


def test_regression_still_folds_when_stratifying_is_meaningless():
    """Continuous targets have no classes to lose, so nothing is skipped there."""
    X, y = make_task(rows=200, columns=4, classification=False)
    assert len(cross_validation_splits(X, y, folds=5, classification=False, seed=11)) == 5


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(oversized="shrink"), "oversized must be one of"),
        (dict(oversized="subsample", folds=1), "folds must be at least 2"),
    ],
)
def test_get_openml_predictions_rejects_bad_sizing(kwargs, message):
    """Bad sizing is caught before anything is downloaded from openml."""
    with pytest.raises(ValueError, match=message):
        get_openml_predictions(model=None, tasks=[], classification=True, **kwargs)


def test_callback_passes_its_sizing_through():
    """The callback used to reach none of these, so a small tabarena was not expressible."""
    callback = OpenMLEvaluationCallback(tasks=[1], oversized="subsample", max_n_samples=500, folds=3, seed=7)

    assert callback.settings == dict(
        max_n_features=500,
        max_n_samples=500,
        oversized="subsample",
        folds=3,
        seed=7,
    )
