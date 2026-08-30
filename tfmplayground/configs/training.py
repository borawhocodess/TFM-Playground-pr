from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    parameters training runs share
    """

    seed: int = 2402
    batch_size: int = 1
    steps: int = 100
    epochs: int = 10000
    grad_clip: float = 1.0


@dataclass
class ClassificationTrainingConfig(TrainingConfig):
    """
    training parameters for classification
    """

    problem: str = "classification"
    lr: float = 1e-4


@dataclass
class RegressionTrainingConfig(TrainingConfig):
    """
    training parameters for regression
    """

    problem: str = "regression"
    lr: float = 1e-4
    criterion: str | None = None
    bucket_borders_min_targets: int = 1_000_000
    bucket_borders_outlier_threshold: float = 10.0


@dataclass
class ExperimentConfig:
    """
    parameters experiment tracking shares
    """

    name: str = "test"
    experiments_dir: str = "workdir/experiments"


@dataclass
class ClassificationExperimentConfig(ExperimentConfig):
    """
    experiment parameters for classification
    """

    problem: str = "classification"


@dataclass
class RegressionExperimentConfig(ExperimentConfig):
    """
    experiment parameters for regression
    """

    problem: str = "regression"
