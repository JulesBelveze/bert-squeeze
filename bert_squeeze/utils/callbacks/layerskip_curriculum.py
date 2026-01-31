from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback


class LayerSkipCurriculumCallback(Callback):
    """Controls which layers contribute to early-exit loss via curriculum learning."""

    def __init__(
        self,
        curriculum_type: str = "rotational",
        rotation_period: Optional[int] = None,
        train_last_layer: bool = True,
    ) -> None:
        super().__init__()
        self.curriculum_type = curriculum_type
        self.rotation_period = rotation_period
        self.train_last_layer = train_last_layer

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
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
            if total_steps is None or total_steps <= 0:
                total_steps = trainer.estimated_stepping_batches
            total_steps = max(1, total_steps)

            enabled_from = max(
                0,
                num_layers - 1 - int(2 * num_layers * trainer.global_step / total_steps),
            )
            mask[enabled_from:] = 1.0
        else:
            mask.fill_(1.0)

        if self.train_last_layer:
            mask[-1] = 1.0

        pl_module.curriculum_mask.copy_(mask)
