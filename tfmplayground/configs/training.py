from dataclasses import dataclass


@dataclass
class TrainingConfig:
    seed: int = 2402
    batch_size: int = 1
    steps: int = 100
    epochs: int = 10000
    lr: float = 1e-4
    grad_clip: float = 1.0
    bucket_borders_min_targets: int = 1_000_000
    bucket_borders_outlier_threshold: float = 10.0


@dataclass
class ExperimentConfig:
    name: str = "test"
    experiments_dir: str = "workdir/experiments"


@dataclass
class ClassificationExperimentConfig(ExperimentConfig):
    problem: str = "classification"


@dataclass
class RegressionExperimentConfig(ExperimentConfig):
    problem: str = "regression"
