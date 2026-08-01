from typing import Dict, List, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig

from ..utils.experiment_logging import ExperimentLogger
from ..utils.optimizers import (
    BertAdam,
    OptimizerParameterGroup,
    build_optimizer_parameter_groups,
    register_legacy_optimizer_state_migration,
)
from ..utils.schedulers import GroupCompatibleReduceLROnPlateau
from ..utils.types import DistillationLoss


class BaseDistiller(pl.LightningModule):
    """
    Base Lightning module to extend to perform distillation.

    Args:
        teacher (Union["pl.LightningModule", "torch.nn.Module"]):
            model to distil knowledge from
        student (Union["pl.LightningModule", "torch.nn.Module"]):
            model to use as a student
        training_config (DictConfig):
            configuration to use for training and to distil the teacher model
        teacher_checkpoint (str):
            path to checkpoints to load to the teacher model
    """

    def __init__(
        self,
        teacher: Union["pl.LightningModule", "torch.nn.Module"],
        student: Union[pl.LightningModule, torch.nn.Module],
        training_config: DictConfig,
        teacher_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__()
        self.params = training_config
        self.teacher = teacher
        self.student = student
        self.teacher_checkpoint = teacher_checkpoint

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def _set_objectives(self) -> None:
        """"""
        raise NotImplementedError()

    def _set_scorers(self) -> None:
        """"""
        raise NotImplementedError()

    def _get_student_parameters(self) -> List[OptimizerParameterGroup]:
        return build_optimizer_parameter_groups(
            self.student.named_parameters(),
            discriminative_learning=self.params.discriminative_learning,
            learning_rates=self.params.learning_rates,
            layer_lr_decay=self.params.get("layer_lr_decay", 1.0),
            weight_decay=self.params.weight_decay,
        )

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        Method to define optimizers and learning rate schedulers

        Returns:
            Tuple[List, List]: a tuple of containing a list of optimizers and
                               a list of schedulers to use during training
        """
        optimizer_parameters = self._get_student_parameters()
        learning_rate = self.params.learning_rates[0]

        if self.params.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimizer_parameters, lr=learning_rate)
        elif self.params.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=learning_rate,
                eps=self.params.adam_eps,
            )
        elif self.params.optimizer == "bertadam":
            optimizer = BertAdam(
                optimizer_parameters,
                lr=learning_rate,
                warmup=self.params.warmup_ratio,
            )
        elif self.params.optimizer == "adam":
            optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
        else:
            raise ValueError(f"Optimizer '{self.params.optimizer}' not supported.")

        if self.params.discriminative_learning:
            register_legacy_optimizer_state_migration(
                optimizer, self.student.named_parameters()
            )

        if self.params.lr_scheduler:
            scheduler = GroupCompatibleReduceLROnPlateau(optimizer)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': 'NeptuneLogger',
                'monitor': 'loss',
            }
            return [optimizer], [lr_scheduler]

        return [optimizer], []

    def training_step(self, batch, _) -> torch.Tensor:
        raise NotImplementedError()

    def test_step(self, batch, _) -> Dict:
        raise NotImplementedError()

    def validation_step(self, batch, _) -> Dict:
        raise NotImplementedError()

    def on_train_epoch_end(self) -> None:
        """"""
        self.s_scorer.reset()

    def on_validation_epoch_end(self) -> None:
        raise NotImplementedError()

    def on_test_epoch_end(self) -> None:
        raise NotImplementedError()

    def loss(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, *args, **kwargs
    ) -> DistillationLoss:
        raise NotImplementedError()

    def log_eval_report(self, *args) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses and metrics.
        """
        results = self.s_valid_scorer.to_dict()
        table = self.s_valid_scorer.get_table(results)
        ExperimentLogger.from_module(self).add_text("eval/report", table)

        # logging losses to neptune
        logging_loss = {
            key: torch.stack(val).mean()
            for key, val in self.s_valid_scorer.losses.items()
        }
        self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})

        # logging other metrics
        for key, value in results.items():
            if not isinstance(value, (list, np.ndarray)):
                self.log_dict({f"eval/{key}": value})
