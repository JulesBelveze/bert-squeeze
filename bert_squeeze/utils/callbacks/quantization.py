import logging
import os
from typing import Any, Iterable, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.callbacks.base import Callback


def custom_trigger_last(trainer: pl.Trainer) -> bool:
    """
    Check if the current epoch is the last one.

    Args:
        trainer (pl.Trainer):
            Lightning trainer
    """
    return trainer.current_epoch == (trainer.max_epochs - 1)


class TransformerQAT(QuantizationAwareTraining):
    """
    Extends `pytorch_lightning.callbacks.QuantizationAwareTraining`

    Quantization allows speeding up inference and decreasing memory requirements
    by performing computations and storing tensors at lower bitwidths (such as INT8
    or FLOAT16) than floating point precision. We use native PyTorch API so for more
    information see PyTorch Quantization.

    Args:
        observer_type (str):
            allows switching between MovingAverageMinMaxObserver as “average”
            (default) and HistogramObserver as “histogram” which is more
            computationally expensive.
        only_last_epoch (bool):
            whether to only perform QAT on the last epoch or not
        layers_to_fuse (List[str]):
            allows you fuse a few layers together as shown in diagram to find which
            layer types can be fused, check https://github.com/pytorch/pytorch/pull/43286.
    """

    def __init__(
        self,
        observer_type: str = "average",
        only_last_epoch: bool = False,
        layers_to_fuse: List[str] = None,
    ):
        super(TransformerQAT, self).__init__(
            qconfig="fbgemm",
            observer_type=observer_type,
            collect_quantization=custom_trigger_last if only_last_epoch else None,
            modules_to_fuse=layers_to_fuse,
        )


class DynamicQuantization(Callback):
    """
    Callback to convert a float model to a dynamic (i.e. weights-only) quantized model at the
    end of training. It uses a PyTorch util function to achieve so ([see here](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_dynamic.html#torch.quantization.quantize_dynamic))

    Args:
        layer_to_quantize (Iterable[str]):
            list of submodule names to apply dynamic quantization to
    """

    def __init__(self, layers_to_quantize: Iterable[str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.layers = set(layers_to_quantize)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        quantized_module = torch.quantization.quantize_dynamic(
            pl_module, self.layers, dtype=torch.qint8
        )

        torch.save(quantized_module.state_dict(), "quantized_model.ckpt")
        logging.info(
            'Model quantized and saved - size (MB):',
            os.path.getsize("quantized_model.ckpt") / 1e6,
        )
