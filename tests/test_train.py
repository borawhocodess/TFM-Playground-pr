import pytest
import torch
from torch import nn

from tfmplayground.configs.models import NanoTabPFNClassifierConfig, NanoTabPFNRegressorConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import Prior
from tfmplayground.training.callbacks import Callback
from tfmplayground.training.train import train
from tfmplayground.utils import ScalarMSELoss, get_default_device


class Quiet(Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.losses.append(loss)

    def close(self):
        pass


class ClassificationPrior(Prior):
    def batch(self, batch_size):
        x = torch.randn(batch_size, 20, 3)
        y = torch.randint(0, 3, (batch_size, 20)).float()
        return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]


class OneTrainRowPrior(Prior):
    def batch(self, batch_size):
        x = torch.randn(batch_size, 6, 3)
        y = torch.randn(batch_size, 6)
        return x[:, :1], y[:, :1], x[:, 1:], y[:, 1:]


class TooManyClassesPrior(Prior):
    def batch(self, batch_size):
        x = torch.randn(batch_size, 20, 3)
        y = torch.zeros(batch_size, 20)
        y[:, -1] = 7
        return x[:, :12], y[:, :12], x[:, 12:], y[:, 12:]


def small(outputs):
    return dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=outputs)


def run(model, prior, criterion, callbacks, epochs=1, steps=2, device=None):
    return train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=epochs,
        batch_size=2,
        steps_per_epoch=steps,
        lr=1e-4,
        grad_clip=1.0,
        device=device if device is not None else get_default_device(),
        callbacks=callbacks,
    )


def test_the_criterion_reaches_the_device():
    model = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(9)))
    criterion = torch.nn.Module()
    criterion.register_buffer("borders", torch.linspace(-3, 3, 10))
    criterion.forward = lambda output, target: (output.mean(-1) - target) ** 2 + criterion.borders.mean()
    run(model, ClassificationPrior(), criterion, [Quiet()])
    assert criterion.borders.device.type == get_default_device()


def test_one_training_row_does_not_poison_the_weights():
    model = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(1)))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(RuntimeError, match="no finite batches"):
        run(model, OneTrainRowPrior(), ScalarMSELoss(), [Quiet()])
    for parameter, was in zip(model.parameters(), before, strict=True):
        assert torch.equal(parameter.detach().cpu(), was.cpu())


def test_a_non_finite_output_is_skipped():
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3)))
    original = model.forward

    def broken(x_train, y_train, x_test):
        return original(x_train, y_train, x_test) * float("nan")

    model.forward = broken
    with pytest.raises(RuntimeError, match="no finite batches"):
        run(model, ClassificationPrior(), nn.CrossEntropyLoss(), [Quiet()])


def test_a_non_finite_loss_is_skipped():
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3)))

    class BrokenLoss(nn.Module):
        def forward(self, output, target):
            return output.sum(-1) * float("inf")

    with pytest.raises(RuntimeError, match="no finite batches"):
        run(model, ClassificationPrior(), BrokenLoss(), [Quiet()])


def test_the_loss_it_returns_is_the_mean_over_the_steps():
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3)))

    class OneLoss(nn.Module):
        def forward(self, output, target):
            return output.sum(-1) * 0.0 + 1.0

    _, loss = run(model, ClassificationPrior(), OneLoss(), [Quiet()], steps=4)
    assert loss == pytest.approx(1.0)


def test_one_training_row_never_reaches_the_model():
    model = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**small(1)))
    original = model.forward
    calls = []

    def counted(x_train, y_train, x_test):
        calls.append(1)
        return original(x_train, y_train, x_test)

    model.forward = counted
    with pytest.raises(RuntimeError, match="no finite batches"):
        run(model, OneTrainRowPrior(), ScalarMSELoss(), [Quiet()])
    assert calls == []


def test_a_non_finite_output_never_reaches_the_criterion():
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3)))
    original = model.forward
    model.forward = lambda a, b, c: original(a, b, c) * float("nan")
    calls = []

    class CountingLoss(nn.Module):
        def forward(self, output, target):
            calls.append(1)
            return nn.functional.cross_entropy(output, target, reduction="none")

    with pytest.raises(RuntimeError, match="no finite batches"):
        run(model, ClassificationPrior(), CountingLoss(), [Quiet()])
    assert calls == []


def test_a_class_the_model_cannot_emit_stops_the_run():
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(3)))
    with pytest.raises(ValueError, match="class 7 but the model has 3 outputs"):
        run(model, TooManyClassesPrior(), nn.CrossEntropyLoss(), [Quiet()])


def test_a_class_the_model_can_emit_is_let_through():
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(**small(8)))
    _, loss = run(model, TooManyClassesPrior(), nn.CrossEntropyLoss(), [Quiet()])
    assert loss > 0
