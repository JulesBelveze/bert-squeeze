import logging
from typing import Dict, Union, Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.base import Callback
from torch.optim import Optimizer

STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]


class ThresholdBasedPruning(Callback):
    def __init__(self, threshold: float, start_pruning_epoch: int = 10):
        self.threshold = threshold
        self.start_pruning_epoch = start_pruning_epoch

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer,
                                 opt_idx: int) -> None:
        """"""
        if pl_module.current_epoch >= self.start_pruning_epoch != -1:
            self._zero_pruned_gradients(pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """"""
        if pl_module.current_epoch >= self.start_pruning_epoch != -1:
            logging.info(
                f"======== Pruning iteration {self.start_pruning_epoch - pl_module.current_epoch - 1} ========")
            self._prune_model(pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when fit ends."""
        if self.start_pruning_epoch == -1:
            self._prune_model(pl_module)

    def _prune_model(self, module):
        """"""
        for name, param in module.named_parameters():
            if "weight" in name:
                param_mask = torch.abs(param) < self.threshold
                param.data[param_mask] = 0.0

    def _zero_pruned_gradients(self, module):
        """Function used for zeroing gradients of pruned weights"""
        for name, param in module.named_parameters():
            if "weight" in name:
                param_data = param.data.cpu()
                param_grad = param.grad.data.cpu()

                dummy_zero_tensor = torch.zeros_like(param_grad)
                param_grad = torch.where(param_data == 0.0, dummy_zero_tensor, param_grad)
                param.grad.data = param_grad.to(self.device)


class SparsityBasedPruning(Callback):
    def __init__(self, sparsity_level: float = 0.0, layers_to_exclude: List[str] = None):
        self.sparsity_level = sparsity_level
        self.layers_to_exclude = layers_to_exclude

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when fit ends."""
        if self.sparsity_level == 0.0:
            return

        # flatten all tensors and create a huge 1D tensor to get the threshold
        flatten_parameters = torch.cat([param.view(-1) for param in pl_module.parameters()], dim=0)

        bottom_k, _ = torch.topk(flatten_parameters.abs(),
                                 int(self.sparsity_level * flatten_parameters.numel()), largest=False,
                                 sorted=True)
        threshold = bottom_k.data[-1]
        self._prune_model(threshold, pl_module)

    @staticmethod
    def _prune_model(threshold, pl_module) -> None:
        """"""
        with torch.no_grad():
            for param in pl_module.parameters():
                mask = torch.lt(torch.abs(param), threshold)
                param.data[mask] = 0.0
