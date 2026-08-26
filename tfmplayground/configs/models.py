from dataclasses import dataclass


@dataclass
class NanoTabPFNModelConfig:
    embedding_size: int = 192
    num_attention_heads: int = 6
    mlp_hidden_size: int = 768
    num_layers: int = 6


@dataclass
class NanoTabPFNClassifierConfig(NanoTabPFNModelConfig):
    num_outputs: int = 10


@dataclass
class NanoTabPFNRegressorConfig(NanoTabPFNModelConfig):
    num_outputs: int = 100
