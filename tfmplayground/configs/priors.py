from dataclasses import dataclass


@dataclass
class ModdedNanoSCMPriorConfig:
    problem: str = "classification"
    min_num_classes: int = 2
    max_num_classes: int = 8
    min_num_cols: int = 20
    max_num_cols: int = 20
    min_num_parent_attempts: int = 3
    max_num_parent_attempts: int = 3
    min_redirection: float = 0.5
    max_redirection: float = 0.5
    min_num_rows: int = 1000
    max_num_rows: int = 1000
    min_num_test_rows: int = 128
    max_num_test_rows: int = 128


@dataclass
class NanoTabICLPriorConfig:
    num_datapoints_max: int = 1000
    num_features: int = 20
    train_fraction_min: float = 0.1
    train_fraction_max: float = 0.9


@dataclass
class NanoTabICLClassificationPriorConfig(NanoTabICLPriorConfig):
    problem: str = "classification"
    max_num_classes: int = 10


@dataclass
class NanoTabICLRegressionPriorConfig(NanoTabICLPriorConfig):
    problem: str = "regression"
    max_num_classes: int = 0


@dataclass
class TabICLPriorConfig:
    num_datapoints_min: int = 128
    num_datapoints_max: int = 1024
    num_features_min: int = 2
    num_features_max: int = 100
    prior_type: str = "mix_scm"
    n_jobs: int = 1


@dataclass
class TabICLClassificationPriorConfig(TabICLPriorConfig):
    problem: str = "classification"
    max_num_classes: int = 10


@dataclass
class TabICLRegressionPriorConfig(TabICLPriorConfig):
    problem: str = "regression"
    max_num_classes: int = 0


@dataclass
class ClassificationPriorDumpConfig:
    problem: str = "classification"
    filename: str = "50x3_3_100k_classification.h5"


@dataclass
class RegressionPriorDumpConfig:
    problem: str = "regression"
    filename: str = "50x3_1280k_regression.h5"
