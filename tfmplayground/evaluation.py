import numpy as np
import openml
import torch
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from tfmplayground.callbacks import Callback
from tfmplayground.interface import TabularClassifier, TabularRegressor

TOY_TASKS_REGRESSION = [
    362443,  # diabetes
]

TOY_TASKS_CLASSIFICATION = [
    59,  # iris
    2382,  # wine
    9946,  # breast_cancer
]

# we hardcode the list here because even if the tasks are cached
# openml.study.get_suite("tabarena-v0.1") might fail if there are connection issues
TABARENA_TASKS = [
    363612,
    363613,
    363614,
    363615,
    363616,
    363618,
    363619,
    363620,
    363621,
    363623,
    363624,
    363625,
    363626,
    363627,
    363628,
    363629,
    363630,
    363631,
    363632,
    363671,
    363672,
    363673,
    363674,
    363675,
    363676,
    363677,
    363678,
    363679,
    363681,
    363682,
    363683,
    363684,
    363685,
    363686,
    363689,
    363691,
    363693,
    363694,
    363696,
    363697,
    363698,
    363699,
    363700,
    363702,
    363704,
    363705,
    363706,
    363707,
    363708,
    363711,
    363712,
]


class OpenMLEvaluationCallback(Callback):
    """
    Evaluates the model on OpenML tasks at the end of each epoch and logs the score to the console,
    roc auc for classification and r2 for regression.

    Takes the same task sizing arguments as get_openml_predictions and passes them straight
    through, so oversized="subsample" gives a small tabarena rather than a short one.
    """

    def __init__(
        self,
        tasks: list[int] | str,
        classification: bool = True,
        device: str | torch.device | None = None,
        max_n_features: int = 500,
        max_n_samples: int = 10_000,
        oversized: str = "skip",
        folds: int | None = None,
        seed: int = 11,
    ):
        self.tasks = tasks
        self.classification = classification
        self.device = device
        self.settings = dict(
            max_n_features=max_n_features,
            max_n_samples=max_n_samples,
            oversized=oversized,
            folds=folds,
            seed=seed,
        )

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        if self.classification:
            wrapped = TabularClassifier(model, self.device)
            predictions = get_openml_predictions(model=wrapped, tasks=self.tasks, **self.settings)
            scores = [roc_auc_score(y_true, y_proba, multi_class="ovr") for y_true, _, y_proba in predictions.values()]
            metric = "avg roc auc"
        else:
            wrapped = TabularRegressor(model, kwargs.get("dist"), self.device)
            predictions = get_openml_predictions(model=wrapped, tasks=self.tasks, **self.settings)
            scores = [r2_score(y_true, y_pred) for y_true, y_pred, _ in predictions.values()]
            metric = "avg r2 score"
        if not scores:
            raise ValueError("OpenML evaluation found no compatible tasks within the configured limits")
        avg_score = sum(scores) / len(scores)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | {metric} {avg_score:.3f}",
            flush=True,
        )
        return {"roc_auc" if self.classification else "r2": avg_score, "tasks_scored": len(scores)}

    def close(self):
        pass


OVERSIZED = ("skip", "subsample")


def can_stratify(y, parts: int) -> bool:
    """Stratifying needs every class to survive the split, which imbalanced tasks do not promise."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return counts.min() >= parts


def shrink_task(X, y, max_n_features: int, max_n_samples: int, classification: bool, seed: int):
    """Cuts an oversized task down to the limits instead of dropping it."""
    if X.shape[1] > max_n_features:
        rng = np.random.default_rng(seed)
        X = X.iloc[:, rng.choice(X.shape[1], size=max_n_features, replace=False)]
    if len(X) > max_n_samples:
        stratify = y if classification and can_stratify(y, 2) else None
        _, X, _, y = train_test_split(X, y, test_size=max_n_samples, stratify=stratify, random_state=seed)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    return X, y


def cross_validation_splits(X, y, folds: int, classification: bool, seed: int):
    """
    Own folds over a shrunk task, because openml's split indices point at the rows we dropped.

    Returns nothing when a classification task cannot be folded safely. Plain KFold splits blindly,
    so a class too rare to stratify can land entirely in one test fold with none of it in that
    fold's training half. The label encoder is fitted on the training half, so it then meets a
    label it has never seen and raises, as does roc auc on a class with no predictions. Measured
    over 200 seeds at 300 rows and five folds: a one row class does this every time, a two row
    class about a quarter of the time. The eval callback runs every epoch, so that is not a bad
    score, it is a training run dying partway through. A skipped task costs one number instead.
    """
    if classification and not can_stratify(y, folds):
        return []
    splitter = StratifiedKFold if classification else KFold
    return list(splitter(n_splits=folds, shuffle=True, random_state=seed).split(X, y))


@torch.no_grad()
def get_openml_predictions(
    *,
    model: TabularRegressor | TabularClassifier,
    tasks: list[int] | str = "tabarena-v0.1",
    max_n_features: int = 500,
    max_n_samples: int = 10_000,
    oversized: str = "skip",
    folds: int | None = None,
    seed: int = 11,
    classification: bool | None = None,
    cache_directory: str | None = None,
):
    """
    Evaluates a model on a set of OpenML tasks and returns predictions.

    Retrieves datasets from OpenML, applies preprocessing, and evaluates the given model on each task.
    Returns true targets, predicted labels, and predicted probabilities for each dataset.

    Args:
        model (TabularRegressor | TabularClassifier):
            A scikit-learn compatible model or classifier to be evaluated.
        tasks (list[int] | str, optional):
            A list of OpenML task IDs or the name of a benchmark suite.
        max_n_features (int, optional):
            Most features a task may carry before oversized decides what to do about it.
        max_n_samples (int, optional):
            Most rows a task may carry before oversized decides what to do about it.
        oversized (str, optional):
            What happens to a task over those limits. "skip" drops it, which shrinks the benchmark.
            "subsample" keeps it and cuts it down to the limits, which shrinks the data instead:
            a seeded random pick of the columns and a stratified sample of the rows.
        folds (int | None, optional):
            How many folds to score each task over. Defaults to 1 when skipping, since that is
            what openml hands us, and 5 when subsampling, where our own folds are all we have and
            cross validation is what buys the signal back from a small sample.
        seed (int, optional):
            Seeds the subsampling and the fold shuffling, so the score moves when the model moves
            and not when the sample does.
        classification (bool | None, optional):
            Whether the model is a classifier (True) or regressor (False). If None, it is inferred from the model type.
        cache_directory (str | None, optional):
            Directory to save OpenML data. If None, default cache path is used.
    Returns:
        dict: A dictionary where keys are dataset names and values are tuples of
            (true targets, predicted labels, predicted probabilities).
    """
    if classification is None:
        classification = isinstance(model, TabularClassifier)  # TODO: change this once we support different models

    if oversized not in OVERSIZED:
        raise ValueError(f"oversized must be one of {sorted(OVERSIZED)}, got {oversized!r}")
    if folds is None:
        folds = 5 if oversized == "subsample" else 1
    if oversized == "subsample" and folds < 2:
        raise ValueError(f"subsampling scores a task over its own folds, so folds must be at least 2, got {folds}")

    if cache_directory is not None:
        set_root_cache_directory(cache_directory)

    if isinstance(tasks, str):
        benchmark_suite = openml.study.get_suite(tasks)
        task_ids = benchmark_suite.tasks
    else:
        task_ids = tasks

    dataset_predictions = {}

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)

        if classification and task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue  # skip task, only classification
        if not classification and task.task_type_id != TaskType.SUPERVISED_REGRESSION:
            continue  # skip task, only regression

        dataset = task.get_dataset(download_data=False)

        n_features = dataset.qualities["NumberOfFeatures"]
        n_samples = dataset.qualities["NumberOfInstances"]
        if oversized == "skip" and (n_features > max_n_features or n_samples > max_n_samples):
            continue  # skip task, too big

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        if oversized == "subsample":
            X, y = shrink_task(X, y, max_n_features, max_n_samples, classification, seed)
            splits = cross_validation_splits(X, y, folds, classification, seed)
            if not splits:
                continue  # a class too rare to fold safely, see cross_validation_splits
        else:
            repeat = 0  # code only supports one repeat
            available = task.get_split_dimensions()[1]
            splits = [
                task.get_train_test_split_indices(fold=fold, repeat=repeat) for fold in range(min(available, folds))
            ]

        targets = []
        predictions = []
        probabilities = []
        for train_indices, test_indices in splits:
            X_train = X.iloc[train_indices].to_numpy()
            y_train = y.iloc[train_indices].to_numpy()
            X_test = X.iloc[test_indices].to_numpy()
            y_test = y.iloc[test_indices].to_numpy()

            if classification:
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)
            targets.append(y_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
            if classification:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # binary classification
                    y_proba = y_proba[:, 1]
                probabilities.append(y_proba)

        y_pred = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0) if len(probabilities) > 0 else None
        dataset_predictions[str(dataset.name)] = (targets, y_pred, probabilities)
    return dataset_predictions
