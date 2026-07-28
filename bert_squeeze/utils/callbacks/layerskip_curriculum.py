from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback


class LayerSkipCurriculumCallback(Callback):
    def __init__(
        self,
        curriculum_type: str = "rotational",
        rotation_period: Optional[int] = None,
    ) -> None:
        super().__init__()
        if curriculum_type not in {"rotational", "gradual", "all"}:
            raise ValueError("curriculum_type must be rotational, gradual, or all.")
        if rotation_period is not None and rotation_period <= 0:
            raise ValueError("rotation_period must be positive.")
        self.curriculum_type = curriculum_type
        self.rotation_period = rotation_period

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        num_layers = getattr(pl_module, "num_layers", 0)
        if num_layers <= 0:
            return

        mask = torch.zeros_like(pl_module.curriculum_mask)
        if self.curriculum_type == "rotational":
            rotation = self.rotation_period or max(1, num_layers - 1)
            for layer_idx in range(num_layers):
                if (trainer.global_step % rotation) == (layer_idx % rotation):
                    mask[layer_idx] = 1.0
        elif self.curriculum_type == "gradual":
            total_steps = trainer.max_steps
            if total_steps <= 0:
                total_steps = int(trainer.estimated_stepping_batches)
            total_steps = max(1, total_steps)

            enabled_layers = min(
                num_layers,
                int(2 * num_layers * trainer.global_step / total_steps),
            )
            enabled_from = num_layers - enabled_layers
            mask[enabled_from:] = 1.0
        elif self.curriculum_type == "all":
            mask.fill_(1.0)

        mask[-1] = 1.0

        pl_module.curriculum_mask.copy_(mask)
