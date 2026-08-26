from dataclasses import dataclass


@dataclass
class TrainingConfig:
    seed: int = 2402
    batch_size: int = 1
    steps: int = 100
    epochs: int = 10000
    lr: float = 1e-4
    grad_clip: float = 1.0
