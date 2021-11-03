import os
from typing import Iterable, List
import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.callbacks.base import Callback


def custom_trigger_last(trainer):
    return trainer.current_epoch == (trainer.max_epochs - 1)


class TransformerQAT(QuantizationAwareTraining):
    def __init__(self, observer_type: str = "average", only_last_epoch: bool = False, layers_to_fuse: List[str] = None):
        super(TransformerQAT, self).__init__(
            qconfig="fbgemm",
            observer_type=observer_type,
            collect_quantization=custom_trigger_last if only_last_epoch else None,
            modules_to_fuse=layers_to_fuse
        )


class DynamicQuantization(Callback):
    def __init__(self, layers_to_quantize: Iterable[str]):
        self.layers = set(layers_to_quantize)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        quantized_module = torch.quantization.quantize_dynamic(
            pl_module, self.layers, dtype=torch.qint8
        )

        torch.save(quantized_module.state_dict(), "quantized_model.ckpt")
        logging.info('Model quantized and saved - size (MB):', os.path.getsize("quantized_model.ckpt") / 1e6)
