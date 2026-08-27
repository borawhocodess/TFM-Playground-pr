from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    max_n_features: int = 100
    max_n_samples: int = 1000
