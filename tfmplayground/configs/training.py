from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 1
    accumulate_gradients: int = 1
    steps: int = 100
    epochs: int = 10000
    lr: float = 1e-4
