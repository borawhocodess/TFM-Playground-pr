from dataclasses import dataclass


@dataclass
class ModdedNanoSCMPriorConfig:
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
    num_test_datapoints: int = 128
    problem: str = "classification"
    max_num_classes: int = 10


@dataclass
class TabICLPriorConfig:
    num_datapoints_min: int = 128
    num_datapoints_max: int = 1024
    num_features_min: int = 2
    num_features_max: int = 100
    problem: str = "classification"
    max_num_classes: int = 10
    prior_type: str = "mix_scm"
    n_jobs: int = 1
