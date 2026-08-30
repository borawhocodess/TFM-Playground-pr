import torch
from tabfm.src.pytorch.model import TabFM
from torch import nn

from tfmplayground.configs.models import TabFMModelConfig
from tfmplayground.models.base import TabularFoundationModel


class TabFMModel(TabFM, TabularFoundationModel):
    def __init__(self, config: TabFMModelConfig) -> None:
        """
        todo
        """
        self.config = config
        super().__init__(
            embed_dim=config.embed_dim,
            max_classes=config.max_classes,
            col_num_blocks=config.col_num_blocks,
            col_nhead=config.col_nhead,
            col_num_inds=config.col_num_inds,
            row_num_blocks=config.row_num_blocks,
            row_nhead=config.row_nhead,
            row_num_cls=config.row_num_cls,
            icl_num_blocks=config.icl_num_blocks,
            icl_nhead=config.icl_nhead,
            ff_factor=config.ff_factor,
            feature_group_size=config.feature_group_size,
            num_freq=config.num_freq,
            decoder_hidden=config.decoder_hidden,
            is_classifier=config.is_classifier,
        )
        fourier_sigma = config.fourier_sigma
        for name in ("fourier_frequencies", "fourier_frequencies_cat"):
            # no jax checkpoint fills these so we make them random parameters
            zeros = getattr(self.cell_embedder, name)
            delattr(self.cell_embedder, name)
            frequencies = torch.randn_like(zeros) * fourier_sigma
            self.cell_embedder.register_parameter(name, nn.Parameter(frequencies))

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        todo
        """
        batch_size, _ = y_train.shape
        _, num_train_rows, _ = X_train.shape
        _, num_test_rows, _ = X_test.shape
        train_size = torch.full([batch_size], num_train_rows, device=X_train.device, dtype=torch.long)
        y_test_zeros = torch.zeros(batch_size, num_test_rows, device=y_train.device, dtype=y_train.dtype)
        x = torch.cat([X_train, X_test], dim=1)
        y = torch.cat([y_train, y_test_zeros], dim=1)
        output = super().forward(x, y, train_size)
        return output[:, num_train_rows:]
