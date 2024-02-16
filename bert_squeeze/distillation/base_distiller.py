from typing import Dict, List, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW

from ..utils.optimizers import BertAdam
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

    def _get_student_parameters(self) -> List[Dict]:
        """
        Method that defines the student's parameters to optimize.

        Returns:
            List[Dict]: group of parameters to optimize
        """
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']

        if self.params.discriminative_learning:
            if (
                isinstance(self.params.learning_rates, ListConfig)
                and len(self.params.learning_rates) > 1
            ):
                groups = [
                    (f'layer.{i}.', self.params.learning_rates[i]) for i in range(12)
                ]
            else:
                lr = (
                    self.params.learning_rates[0]
                    if isinstance(self.params.learning_rates, ListConfig)
                    else self.params.learning_rates
                )
                groups = [
                    (f'layer.{i}.', lr * pow(self.params.layer_lr_decay, 11 - i))
                    for i in range(12)
                ]

            group_all = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters, decay_optimizer_parameters = [], []
            for g, l in groups:
                no_decay_optimizer_parameters.append(
                    {
                        'params': [
                            p
                            for n, p in self.student.named_parameters()
                            if not any(nd in n for nd in no_decay)
                            and any(nd in n for nd in [g])
                        ],
                        'weight_decay_rate': self.params.weight_decay,
                        'lr': l,
                    }
                )
                decay_optimizer_parameters.append(
                    {
                        'params': [
                            p
                            for n, p in self.student.named_parameters()
                            if any(nd in n for nd in no_decay)
                            and any(nd in n for nd in [g])
                        ],
                        'weight_decay_rate': 0.0,
                        'lr': l,
                    }
                )

            group_all_parameters = [
                {
                    'params': [
                        p
                        for n, p in self.student.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay_rate': self.params.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.student.named_parameters()
                        if any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay_rate': 0.0,
                },
            ]
            optimizer_grouped_parameters = (
                no_decay_optimizer_parameters
                + decay_optimizer_parameters
                + group_all_parameters
            )
        else:
            optimizer_grouped_parameters = [
                {
                    'params': [
                        p
                        for n, p in self.student.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    'weight_decay_rate': self.params.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.student.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    'weight_decay_rate': 0.0,
                },
            ]
        return optimizer_grouped_parameters

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        Method to define optimizers and learning rate schedulers

        Returns:
            Tuple[List, List]: a tuple of containing a list of optimizers and
                               a list of schedulers to use during training
        """
        optimizer_parameters = self._get_student_parameters()
        if self.params.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_parameters, lr=self.params.learning_rates[0]
            )
        elif self.params.optimizer == "adamw":
            optimizer = AdamW(
                optimizer_parameters,
                lr=self.params.learning_rates[0],
                eps=self.params.adam_eps,
            )
        elif self.params.optimizer == "bertadam":
            optimizer = BertAdam(
                optimizer_parameters,
                lr=self.params.learning_rates[0],
                warmup=self.params.warmup_ratio,
            )
        elif self.params.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_parameters, lr=self.params.learning_rates[0]
            )
        else:
            raise ValueError(f"Optimizer '{self.params.optimizer}' not supported.")

        if self.params.lr_scheduler:
            scheduler = ReduceLROnPlateau(optimizer)
            lr_scheduler = {
                "scheduler": scheduler,
                "name": "NeptuneLogger",
                "monitor": "loss",
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
        table = self.s_valid_scorer.get_table()
        self.logger.experiment.add_text("eval/report", table)

        # logging losses to neptune
        logging_loss = {
            key: torch.stack(val).mean()
            for key, val in self.s_valid_scorer.losses.items()
        }
        self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})

        # logging other metrics
        eval_report = self.s_valid_scorer.to_dict()
        for key, value in eval_report.items():
            if not isinstance(value, list) and not isinstance(value, np.ndarray):
                self.log_dict(
                    {f"eval/loss_{key}": val for key, val in logging_loss.items()}
                )
