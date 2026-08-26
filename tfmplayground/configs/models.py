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


@dataclass
class TabICLModelConfig:
    num_quantiles: int = 999
    embed_dim: int = 128
    col_num_blocks: int = 3
    col_nhead: int = 8
    col_num_inds: int = 128
    col_affine: bool = False
    col_feature_group: str = "same"
    col_feature_group_size: int = 3
    col_target_aware: bool = True
    col_ssmax: str = "qassmax-mlp-elementwise"
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    row_rope_base: float = 100000
    row_rope_interleaved: bool = True
    icl_num_blocks: int = 12
    icl_nhead: int = 8
    icl_ssmax: str = "qassmax-mlp-elementwise"
    ff_factor: int = 2
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True
    bias_free_ln: bool = False
    zero_init: bool = True
    recompute: bool = False


@dataclass
class TabICLClassifierConfig(TabICLModelConfig):
    max_classes: int = 10


@dataclass
class TabICLRegressorConfig(TabICLModelConfig):
    max_classes: int = 0
