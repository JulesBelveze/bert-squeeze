from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

__all__ = ["DynamicQuantization"]

QuantizationTarget = Union[str, type[nn.Module]]


class DynamicQuantization(Callback):
    """Export a dynamically quantized version of the trained model."""

    def __init__(
        self,
        layers_to_quantize: Optional[Iterable[QuantizationTarget]] = None,
        output_path: Union[str, Path] = "quantized_model.ckpt",
    ) -> None:
        super().__init__()
        self.layers = _quantization_targets(layers_to_quantize)
        self.output_path = Path(output_path)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        target = _quantization_model(pl_module)
        _validate_module_names(target, self.layers)
        _configure_quantized_engine()
        model_to_quantize = deepcopy(target).cpu()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_to_quantize,
            self.layers,
            dtype=torch.qint8,
            inplace=True,
        )
        quantized_model.eval()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model, self.output_path)
        logging.info(
            "Saved dynamically quantized model to %s (%.2f MB).",
            self.output_path,
            self.output_path.stat().st_size / 1e6,
        )


def _quantization_targets(
    layers_to_quantize: Optional[Iterable[QuantizationTarget]],
) -> set[QuantizationTarget]:
    if layers_to_quantize is None:
        return {nn.Linear}

    layers = set(layers_to_quantize)
    if not layers:
        raise ValueError(
            "layers_to_quantize must contain at least one module name or type."
        )
    if not all(
        isinstance(layer, str)
        or (isinstance(layer, type) and issubclass(layer, nn.Module))
        for layer in layers
    ):
        raise TypeError(
            "layers_to_quantize must contain module names or nn.Module types."
        )
    return layers


def _quantization_model(module: pl.LightningModule) -> nn.Module:
    student = getattr(module, "student", None)
    candidate = student if isinstance(student, nn.Module) else module
    model = getattr(candidate, "model", None)
    return model if isinstance(model, nn.Module) else candidate


def _validate_module_names(
    model: nn.Module,
    layers: set[QuantizationTarget],
) -> None:
    module_names = {layer for layer in layers if isinstance(layer, str)}
    unknown_names = module_names - set(dict(model.named_modules()))
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise ValueError(f"Unknown module names for dynamic quantization: {names}.")


def _configure_quantized_engine() -> None:
    if torch.backends.quantized.engine != "none":
        return

    supported_engines = set(torch.backends.quantized.supported_engines)
    for engine in ("x86", "fbgemm", "qnnpack", "onednn"):
        if engine in supported_engines:
            torch.backends.quantized.engine = engine
            return
    raise RuntimeError("This PyTorch build does not support dynamic quantization.")
