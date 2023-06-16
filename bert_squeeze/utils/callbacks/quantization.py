import logging
import os
from typing import Any, Iterable

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback


class DynamicQuantization(Callback):
    """
    Callback to convert a float model to a dynamic (i.e. weights-only) quantized model at the
    end of training. It uses a PyTorch util function to achieve so ([see here](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_dynamic.html#torch.quantization.quantize_dynamic))

    Args:
        layer_to_quantize (Iterable[str]):
            list of submodule names to apply dynamic quantization to
    """

    def __init__(
        self, layers_to_quantize: Iterable[str] = None, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.layers = set(layers_to_quantize) if layers_to_quantize is not None else None

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """"""
        if self.layers is None:
            self.layers = set([layer for layer, _ in pl_module.named_parameters()])

        quantized_module = torch.quantization.quantize_dynamic(
            pl_module, self.layers, dtype=torch.qint8
        )

        torch.save(quantized_module.state_dict(), "quantized_model.ckpt")
        logging.info(
            f'Model quantized and saved - size (MB): {os.path.getsize("quantized_model.ckpt") / 1e6}'
        )
