from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """RMSNorm that upcasts to float32 for the norm step to avoid bfloat16 instability."""
    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            return super().forward(x.float()).to(x.dtype)
        return super().forward(x)


class ThinkingRows(nn.Module):
    """Prepends N learnable row tokens before the data, letting the model accumulate
    intermediate representations before attending to test rows."""
    def __init__(self, num_thinking_rows: int, embedding_size: int):
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        if num_thinking_rows > 0:
            self.row_tokens = nn.Parameter(torch.empty(num_thinking_rows, embedding_size))
            nn.init.normal_(self.row_tokens)

    def forward(self, x: torch.Tensor, single_eval_pos: int):
        if self.num_thinking_rows == 0:
            return x, single_eval_pos
        b, r, c, e = x.shape
        thinking = self.row_tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, e)
        x = torch.cat([thinking, x], dim=1)
        return x, single_eval_pos + self.num_thinking_rows


class NanoTabPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int,
                 num_layers: int, num_outputs: int, residual_decay: float = 1.0,
                 num_thinking_rows: int = 0):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.residual_decay = residual_decay
        self.num_thinking_rows = num_thinking_rows

        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.thinking_rows = ThinkingRows(num_thinking_rows, embedding_size)
        self.transformer_encoder = TransformerEncoderStack(
            num_layers, embedding_size, num_attention_heads, mlp_hidden_size,
            residual_decay=residual_decay,
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

        # bucket borders baked into the model so checkpoints are self-contained
        self.register_buffer("borders", None, persistent=True)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int,
                 num_mem_chunks: int = 1) -> torch.Tensor:
        # num_mem_chunks kept for API compat but no longer used (replaced by torch.compile)
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        src, single_eval_pos = self.thinking_rows(src, single_eval_pos)
        output = self.transformer_encoder(src, single_eval_pos)
        output = output[:, single_eval_pos:, -1, :]
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = x[:, :single_eval_pos].mean(dim=1, keepdim=True)
        std = x[:, :single_eval_pos].std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = y_train.mean(dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int,
                 mlp_hidden_size: int, residual_decay: float = 1.0):
        super().__init__()
        self.residual_decay = residual_decay
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, single_eval_position: int) -> torch.Tensor:
        for i, block in enumerate(self.transformer_blocks):
            if self.residual_decay != 1.0:
                x = x * (self.residual_decay ** i)
            x = block(x, single_eval_position)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        assert embedding_size % nhead == 0, "embedding_size must be divisible by nhead"
        self.num_heads = nhead
        self.head_dim = embedding_size // nhead

        factory_kwargs = {'device': device, 'dtype': dtype}
        # fused QKV projection — one matrix instead of three; enables batched Muon update
        self.qkv_features = nn.Linear(embedding_size, 3 * embedding_size, **factory_kwargs)
        self.qkv_datapoints = nn.Linear(embedding_size, 3 * embedding_size, **factory_kwargs)

        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size, **factory_kwargs)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size, **factory_kwargs)

        self.norm1 = LowerPrecisionRMSNorm(embedding_size, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LowerPrecisionRMSNorm(embedding_size, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LowerPrecisionRMSNorm(embedding_size, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, src: torch.Tensor, single_eval_position: int) -> torch.Tensor:
        b, r, c, e = src.shape
        h, hd = self.num_heads, self.head_dim

        # --- feature attention: each row attends over its columns ---
        x = src.reshape(b * r, c, e)
        qkv = self.qkv_features(x).reshape(b * r, c, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b * r, c, e)
        src = (x + src.reshape(b * r, c, e)).reshape(b, r, c, e)
        src = self.norm1(src)

        # --- datapoint attention: each column attends over its rows ---
        # test rows attend to train rows only; train rows attend to each other
        x = src.transpose(1, 2).reshape(b * c, r, e)
        qkv = self.qkv_datapoints(x).reshape(b * c, r, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q_left, q_right = q.split([single_eval_position, r - single_eval_position], dim=2)
        k_train = k[:, :, :single_eval_position, :]
        v_train = v[:, :, :single_eval_position, :]

        x_left = F.scaled_dot_product_attention(q_left, k_train, v_train)
        x_right = F.scaled_dot_product_attention(q_right, k_train, v_train)
        x = torch.cat([x_left, x_right], dim=2).transpose(1, 2).reshape(b * c, r, e)

        src = (x + src.transpose(1, 2).reshape(b * c, r, e)).reshape(b, c, r, e).transpose(2, 1)
        src = self.norm2(src)

        # --- MLP ---
        src = src + self.linear2(F.gelu(self.linear1(src)))
        src = self.norm3(src)

        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Legacy architecture (v1): separate q/k/v projections + out_proj + LayerNorm
# Used to load checkpoints trained before the fused-QKV / RMSNorm rewrite.
# ---------------------------------------------------------------------------

class _LegacyAttentionProjections(nn.Module):
    """Holds the four projection matrices; forward is handled by the parent layer."""
    def __init__(self, embedding_size: int):
        super().__init__()
        self.q_proj   = nn.Linear(embedding_size, embedding_size)
        self.k_proj   = nn.Linear(embedding_size, embedding_size)
        self.v_proj   = nn.Linear(embedding_size, embedding_size)
        self.out_proj  = nn.Linear(embedding_size, embedding_size)


class _LegacyTransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        assert embedding_size % nhead == 0
        self.num_heads = nhead
        self.head_dim  = embedding_size // nhead

        self.self_attention_between_features   = _LegacyAttentionProjections(embedding_size)
        self.self_attention_between_datapoints = _LegacyAttentionProjections(embedding_size)

        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size)

        self.norm1 = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embedding_size, eps=layer_norm_eps)

    def _attn(self, proj: _LegacyAttentionProjections,
              q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        b, sq, e = q_in.shape
        sk       = k_in.shape[1]
        h, hd    = self.num_heads, self.head_dim
        q = proj.q_proj(q_in).reshape(b, sq, h, hd).transpose(1, 2)
        k = proj.k_proj(k_in).reshape(b, sk, h, hd).transpose(1, 2)
        v = proj.v_proj(v_in).reshape(b, sk, h, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, sq, e)
        return proj.out_proj(out)

    def forward(self, src: torch.Tensor, single_eval_position: int) -> torch.Tensor:
        b, r, c, e = src.shape
        h, hd = self.num_heads, self.head_dim

        # feature attention
        x   = src.reshape(b * r, c, e)
        out = self._attn(self.self_attention_between_features, x, x, x)
        src = self.norm1((out + x).reshape(b, r, c, e))

        # datapoint attention (test rows attend to train only)
        x   = src.transpose(1, 2).reshape(b * c, r, e)
        attn = self.self_attention_between_datapoints
        sep  = single_eval_position

        q_full = attn.q_proj(x).reshape(b * c, r, h, hd).transpose(1, 2)
        k_full = attn.k_proj(x).reshape(b * c, r, h, hd).transpose(1, 2)
        v_full = attn.v_proj(x).reshape(b * c, r, h, hd).transpose(1, 2)

        k_train = k_full[:, :, :sep]
        v_train = v_full[:, :, :sep]
        q_left, q_right = q_full.split([sep, r - sep], dim=2)

        out_left  = F.scaled_dot_product_attention(q_left,  k_train, v_train)
        out_right = F.scaled_dot_product_attention(q_right, k_train, v_train)
        out = torch.cat([out_left, out_right], dim=2).transpose(1, 2).reshape(b * c, r, e)
        out = attn.out_proj(out)
        src = self.norm2((out + x).reshape(b, c, r, e).transpose(2, 1))

        # MLP
        src = self.norm3(src + self.linear2(F.gelu(self.linear1(src))))
        return src


class _LegacyTransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, nhead: int, mlp_hidden_size: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            _LegacyTransformerEncoderLayer(embedding_size, nhead, mlp_hidden_size)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, single_eval_position: int) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, single_eval_position)
        return x


class NanoTabPFNModelV1(nn.Module):
    """
    Legacy model architecture (before fused-QKV / RMSNorm rewrite).
    Use this to load checkpoints trained with the old code.
    """
    def __init__(self, embedding_size: int, num_attention_heads: int,
                 mlp_hidden_size: int, num_layers: int, num_outputs: int, **_):
        super().__init__()
        self.embedding_size      = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size     = mlp_hidden_size
        self.num_layers          = num_layers
        self.num_outputs         = num_outputs

        self.feature_encoder  = FeatureEncoder(embedding_size)
        self.target_encoder   = TargetEncoder(embedding_size)
        self.transformer_encoder = _LegacyTransformerEncoderStack(
            num_layers, embedding_size, num_attention_heads, mlp_hidden_size)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

        self.register_buffer("borders", None, persistent=True)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int,
                 num_mem_chunks: int = 1) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        output = self.transformer_encoder(src, single_eval_pos)
        output = output[:, single_eval_pos:, -1, :]
        return self.decoder(output)
