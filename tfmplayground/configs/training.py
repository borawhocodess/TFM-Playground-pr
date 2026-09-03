from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    settings training runs share
    """

    seed: int = 2402
    batch_size: int = 1
    steps: int = 100
    epochs: int = 10000
    grad_clip: float = 1.0


@dataclass
class ClassificationTrainingConfig(TrainingConfig):
    """
    training settings for classification
    """

    problem: str = "classification"
    lr: float = 1e-4


@dataclass
class RegressionTrainingConfig(TrainingConfig):
    """
    training settings for regression
    """

    problem: str = "regression"
    lr: float = 1e-4
    criterion: str | None = None
    bucket_borders_min_targets: int = 1_000_000


@dataclass
class ExperimentConfig:
    """
    settings experiment tracking shares
    """

    name: str = "test"
    experiments_dir: str = "workdir/experiments"


@dataclass
class ClassificationExperimentConfig(ExperimentConfig):
    """
    experiment settings for classification
    """

    problem: str = "classification"


@dataclass
class RegressionExperimentConfig(ExperimentConfig):
    """
    experiment settings for regression
    """

    problem: str = "regression"
