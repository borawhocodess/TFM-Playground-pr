from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """
    parameters that evaluation processes use
    """

    tasks: str | list = "toy"
    max_n_features: int = 100
    max_n_samples: int = 1000
