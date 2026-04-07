import torch
from torch import nn
import time
from torch.utils.data import DataLoader
from typing import Dict
from pfns.bar_distribution import FullSupportBarDistribution
import schedulefree
import os

from tfmplayground.callbacks import Callback
from tfmplayground.losses import QuantileLoss
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.muon import SingleDeviceMuonWithAuxAdam
from tfmplayground.utils import get_default_device

GRAD_CLIP = 0.3

torch.set_float32_matmul_precision('high')


def _build_muon_optimizer(model: NanoTabPFNModel, lr: float, muon_lr: float):
    """
    Split parameters into two groups:
    - Muon:  2-D weight matrices inside the transformer (hidden layers only)
    - AdamW: everything else — encoders, decoder, biases, norms
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model
    transformer_param_ids = {id(p) for p in raw.transformer_encoder.parameters()}

    muon_params, adam_params = [], []
    for p in model.parameters():
        if id(p) in transformer_param_ids and p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    param_groups = [
        dict(params=muon_params, lr=muon_lr, momentum=0.95, weight_decay=0.0, use_muon=True),
        dict(params=adam_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, use_muon=False),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
    epochs: int,
    accumulate_gradients: int = 1,
    lr: float = 1e-3,
    device: torch.device = None,
    callbacks: list[Callback] = None,
    ckpt: Dict[str, torch.Tensor] = None,
    multi_gpu: bool = False,
    run_name: str = 'nanoTFM',
    experiment_id: str = None,
    experiment_dir: str = None,
    mixed_precision: bool = True,
    warmup_steps: int = 1000,
    optimizer_type: str = 'schedulefree',
    muon_lr: float = 0.02,
    cosine_decay: bool = True,
    patience: int = 100,
):
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)

    if optimizer_type == 'muon':
        optimizer = _build_muon_optimizer(model, lr=lr, muon_lr=muon_lr)
    else:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.1, warmup_steps=warmup_steps)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = None
    if cosine_decay and optimizer_type != 'schedulefree':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    quantile_task = isinstance(criterion, QuantileLoss)
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task and not quantile_task

    device = torch.device(device) if isinstance(device, str) else device
    use_amp = mixed_precision and device.type == 'cuda'
    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type='cpu', enabled=False)
    # bfloat16 has the same exponent range as float32 — no GradScaler needed

    assert prior.num_steps % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'

    init_param_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
    best_score = float('-inf')
    epochs_without_improvement = 0

    try:
        for epoch in range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1):
            curriculum_str = ""
            if hasattr(prior, 'on_epoch_start'):
                prior.on_epoch_start(epoch, epochs)
                curriculum_str = f" | curriculum f={prior.cur_features} r={prior.cur_rows}"
            epoch_start_time = time.time()
            model.train()
            if hasattr(optimizer, 'train'):
                optimizer.train()
            total_loss = 0.
            total_grad_norm = 0.
            skipped_steps = 0
            n_optimizer_steps = 0
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data['single_eval_pos']
                data = (full_data['x'].to(device),
                        full_data['y'][:, :single_eval_pos].to(device))
                if torch.isnan(data[0]).any() or torch.isnan(data[1]).any():
                    continue
                targets = full_data['target_y'].to(device)

                if regression_task or quantile_task:
                    y_mean = data[1].mean(dim=1, keepdim=True)
                    y_std = data[1].std(dim=1, keepdim=True) + 1e-8
                    y_norm = (data[1] - y_mean) / y_std
                    data = (data[0], y_norm)

                with autocast_ctx:
                    output = model(data, single_eval_pos=single_eval_pos)
                    targets = targets[:, single_eval_pos:]
                    if regression_task or quantile_task:
                        targets = (targets - y_mean) / y_std
                    if classification_task:
                        targets = targets.reshape((-1,)).to(torch.long)
                        output = output.view(-1, output.shape[-1])
                    losses = criterion(output, targets)
                    loss = losses.mean() / accumulate_gradients

                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients

                if (i + 1) % accumulate_gradients == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
                    total_grad_norm += grad_norm
                    grads_finite = all(
                        p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                    if grads_finite:
                        optimizer.step()
                        n_optimizer_steps += 1
                    else:
                        skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            update_steps = prior.num_steps // accumulate_gradients
            mean_grad_norm = total_grad_norm / max(update_steps, 1)
            param_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
            param_drift = param_norm - init_param_norm

            diag_parts = [f"grad_norm {mean_grad_norm:.3f}", f"param_drift {param_drift:+.3f}"]
            if skipped_steps:
                diag_parts.append(f"SKIPPED {skipped_steps}/{update_steps} steps (inf/nan grads)")
            diag_str = " | ".join(diag_parts) + curriculum_str

            model.eval()
            if hasattr(optimizer, 'eval'):
                optimizer.eval()

            if scheduler is not None:
                scheduler.step()

            raw_model = model.module if multi_gpu else model
            training_state = {
                'epoch': epoch,
                'architecture': {
                    'num_layers': int(raw_model.num_layers),
                    'embedding_size': int(raw_model.embedding_size),
                    'num_attention_heads': int(raw_model.num_attention_heads),
                    'mlp_hidden_size': int(raw_model.mlp_hidden_size),
                    'num_outputs': int(raw_model.num_outputs),
                    'residual_decay': float(raw_model.residual_decay),
                    'num_thinking_rows': int(raw_model.num_thinking_rows),
                    'use_qassmax': bool(raw_model.use_qassmax),
                    'use_quantile_loss': bool(raw_model.use_quantile_loss),
                },
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            task_name = 'classifier' if classification_task else 'regressor'
            torch.save(training_state, os.path.join(experiment_dir, f'{experiment_id}-{task_name}-checkpoint.pth'))

            epoch_score = float('-inf')
            for callback in callbacks:
                if isinstance(criterion, FullSupportBarDistribution):
                    result = callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, raw_model, dist=criterion, diag=diag_str)
                else:
                    result = callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, raw_model, diag=diag_str)
                if isinstance(result, (int, float)) and result > epoch_score:
                    epoch_score = result

            if epoch_score > best_score:
                best_score = epoch_score
                epochs_without_improvement = 0
                torch.save(training_state, os.path.join(experiment_dir, f'{experiment_id}-{task_name}-best.pth'))
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs, best score {best_score:.4f})")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), total_loss
