import logging
from typing import Dict, Union, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.base import Callback
from torch.optim import Optimizer

STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]


class ThresholdBasedPruning(Callback):
    def __init__(self, threshold: float, start_pruning_epoch: int = 10):
        self.threshold = threshold
        self.start_pruning_epoch = start_pruning_epoch

    def _prune_model(self):
        """"""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_mask = torch.abs(param) < self.threshold
                param.data[param_mask] = 0.0

    def _zero_pruned_gradients(self):
        """Function used for zeroing gradients of pruned weights"""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_data = param.data.cpu()
                param_grad = param.grad.data.cpu()

                dummy_zero_tensor = torch.zeros_like(param_grad)
                param_grad = torch.where(param_data == 0.0, dummy_zero_tensor, param_grad)
                param.grad.data = torch.from_numpy(param_grad).to(self.device)

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer,
                                 opt_idx: int) -> None:
        """"""
        if pl_module.current_epoch >= self.start_pruning_epoch != -1:
            self._zero_pruned_gradients()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """"""
        if pl_module.current_epoch >= self.start_pruning_epoch != -1:
            logging.info(
                f"======== Pruning iteration {self.start_pruning_epoch - pl_module.current_epoch - 1} ========")
            self._prune_model()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when fit ends."""
        if self.start_pruning_epoch == -1:
            self._prune_model()
