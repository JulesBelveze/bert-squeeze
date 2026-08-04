from __future__ import annotations

from pathlib import Path
from typing import Iterator

import lightning.pytorch as pl
import pytest
import torch
from torch import nn

from bert_squeeze.utils.callbacks.quantization import DynamicQuantization


class _ModelModule(pl.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


class _DistillerModule(pl.LightningModule):
    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student


class _TwoLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(inputs)))


@pytest.fixture(autouse=True)
def _restore_quantized_engine() -> Iterator[None]:
    engine = torch.backends.quantized.engine
    yield
    torch.backends.quantized.engine = engine


def _load_quantized_model(path: Path) -> nn.Module:
    return torch.load(path, weights_only=False)


def test_dynamic_quantization_exports_the_underlying_model(tmp_path: Path) -> None:
    callback = DynamicQuantization(output_path=tmp_path / "model.ckpt")
    module = _ModelModule(nn.Sequential(nn.Linear(4, 2)))

    callback.on_fit_end(None, module)

    quantized_model = _load_quantized_model(callback.output_path)
    assert isinstance(module.model[0], nn.Linear)
    assert isinstance(quantized_model[0], nn.quantized.dynamic.Linear)
    assert quantized_model(torch.ones(2, 4)).shape == (2, 2)


def test_dynamic_quantization_targets_a_distillation_student(tmp_path: Path) -> None:
    callback = DynamicQuantization(output_path=tmp_path / "student.ckpt")
    module = _DistillerModule(nn.Sequential(nn.Linear(4, 2)))

    callback.on_fit_end(None, module)

    quantized_model = _load_quantized_model(callback.output_path)
    assert isinstance(quantized_model, nn.Sequential)
    assert isinstance(quantized_model[0], nn.quantized.dynamic.Linear)


def test_dynamic_quantization_accepts_explicit_module_names(tmp_path: Path) -> None:
    callback = DynamicQuantization(
        layers_to_quantize=["first"],
        output_path=tmp_path / "model.ckpt",
    )
    module = _ModelModule(_TwoLinearModel())

    callback.on_fit_end(None, module)

    quantized_model = _load_quantized_model(callback.output_path)
    assert isinstance(quantized_model.first, nn.quantized.dynamic.Linear)
    assert isinstance(quantized_model.second, nn.Linear)


def test_dynamic_quantization_rejects_unknown_module_names(tmp_path: Path) -> None:
    callback = DynamicQuantization(
        layers_to_quantize=["missing"],
        output_path=tmp_path / "model.ckpt",
    )
    module = _ModelModule(nn.Sequential(nn.Linear(4, 2)))

    with pytest.raises(
        ValueError,
        match="Unknown module names for dynamic quantization: missing.",
    ):
        callback.on_fit_end(None, module)
