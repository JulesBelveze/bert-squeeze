import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from ..utils.losses import LabelSmoothingLoss
from ..utils.optimizers import BertAdam
from ..utils.scorers import BaseSequenceClassificationScorer, LMScorer, Scorer
from ..utils.types import Loss


class BaseTransformerModule(pl.LightningModule):
    """
    Base class to extend for all Transformer-based modules.

    Args:
        training_config (DictConfig):
            training configuration
        pretrained_model (str):
            name of the pretrained Transformer model to use
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        scorer (Scorer):
            helper object to compute performance metrics during training
    """

    BASE_CLASS_MODEL = None

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        super().__init__()

        self.config = training_config

        self.pretrained_model = pretrained_model
        self.model = (
            self.BASE_CLASS_MODEL.from_pretrained(self.pretrained_model)
            if model is None
            else model
        )
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []

        self._set_scorers(scorer)

    def forward(self, **kwargs):
        """"""
        raise NotImplementedError()

    def training_step(self, batch, batch_idx, *args, **kwargs):
        """"""
        raise NotImplementedError()

    def on_train_epoch_end(self) -> None:
        """"""
        self.scorer.reset()
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """"""
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        """"""
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """"""
        raise NotImplementedError()

    def on_test_epoch_end(self) -> None:
        """"""
        self.log("Test results", self.test_scorer.get_table())
        self.test_scorer.reset()
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        Method to define optimizers and learning rate schedulers

        Returns:
            Tuple[List, List]: a tuple of containing a list of optimizers and
                               a list of schedulers to use during training
        """
        optimizer_parameters = self._get_optimizer_parameters()

        optimizer_name = self.config.get("optimizer", "adamw")
        if optimizer_name == "adamw":
            optimizer = AdamW(
                optimizer_parameters,
                lr=(
                    self.config.learning_rates[0]
                    if isinstance(self.config.learning_rates, ListConfig)
                    else self.config.learning_rate
                ),
                eps=self.config.adam_eps,
            )

            if self.config.lr_scheduler:
                scheduler = ReduceLROnPlateau(optimizer)
                lr_scheduler = {"scheduler": scheduler, "name": "NeptuneLogger"}
                return [optimizer], [lr_scheduler]

        elif optimizer_name == "bertadam":
            optimizer = BertAdam(
                optimizer_parameters,
                lr=(
                    self.config.learning_rates[0]
                    if isinstance(self.config.learning_rates, ListConfig)
                    else self.config.learning_rate
                ),
                warmup=self.config.warmup_ratio,
            )

        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=(
                    self.config.learning_rates[0]
                    if isinstance(self.config.learning_rates, ListConfig)
                    else self.config.learning_rate
                ),
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_parameters,
                lr=(
                    self.config.learning_rates[0]
                    if isinstance(self.config.learning_rates, ListConfig)
                    else self.config.learning_rate
                ),
            )
        else:
            raise ValueError(f"Optimizer '{self.config.optimizer}' not supported.")

        return [optimizer], []

    def _set_objective(self) -> None:
        """"""
        raise NotImplementedError()

    @staticmethod
    def _sanity_checks(training_config: DictConfig) -> None:
        """
        Args:
            training_config (DictConfig):
                training configuration
        """
        assert (
            training_config.logging_steps > 0
        ), "'logging_steps' should be strictly greater than 0"
        assert (
            training_config.logging_steps > training_config.accumulation_steps
        ), "'logging_steps' should be greater than 'accumulation_steps'"

    def _get_optimizer_parameters(self) -> List[Dict]:
        """
        Method that defines the parameters to optimize.

        Returns:
            List[Dict]: group of parameters to optimize
        """
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']

        if self.config.discriminative_learning:
            if (
                isinstance(self.config.learning_rates, ListConfig)
                and len(self.config.learning_rates) > 1
            ):
                groups = [
                    (f'layer.{i}.', self.config.learning_rates[i]) for i in range(12)
                ]
            else:
                lr = (
                    self.config.learning_rates[0]
                    if isinstance(self.config.learning_rates, ListConfig)
                    else self.config.learning_rates
                )
                groups = [
                    (f'layer.{i}.', lr * pow(self.config.layer_lr_decay, 11 - i))
                    for i in range(12)
                ]

            group_all = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters, decay_optimizer_parameters = [], []
            for g, l in groups:
                no_decay_optimizer_parameters.append(
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if not any(nd in n for nd in no_decay)
                            and any(nd in n for nd in [g])
                        ],
                        'weight_decay_rate': self.config.weight_decay,
                        'lr': l,
                    }
                )
                decay_optimizer_parameters.append(
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
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
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay_rate': self.config.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
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
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    'weight_decay_rate': self.config.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    'weight_decay_rate': 0.0,
                },
            ]
        return optimizer_grouped_parameters

    def _set_scorers(self, *args, **kwargs) -> None:
        """"""
        raise NotImplementedError()

    def freeze_encoder(self) -> None:
        """Freeze encoder layers"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder layers"""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def loss(self, *args, **kwargs) -> Loss:
        """"""
        raise NotImplementedError()

    def log_eval_report(self, *args) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses and metrics
        """
        table = self.valid_scorer.get_table()
        self.logger.experiment.add_text("eval/report", table)

        logging_loss = {
            key: torch.stack(val).mean() for key, val in self.valid_scorer.losses.items()
        }
        self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})

        eval_report = self.valid_scorer.to_dict()
        for metric, value in eval_report.items():
            if isinstance(value, dict):
                self.log_dict({f"eval/{metric}/{key}": v for key, v in value.items()})
            elif not isinstance(value, list) and not isinstance(value, np.ndarray):
                self.log("eval/{}".format(metric), value)


class BaseSequenceClassificationTransformerModule(BaseTransformerModule):
    """
    Base class to extend for Transformer based sequence classification tasks.

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels
        pretrained_model (str):
            name of the pretrained Transformer model to use
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        scorer (Scorer):
            helper object to compute performance metrics during training
    """

    BASE_CLASS_MODEL = AutoModelForSequenceClassification

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        super().__init__(training_config, pretrained_model, model, scorer, **kwargs)
        self._sanity_checks(training_config)

        self.num_labels = num_labels
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model, num_labels=num_labels
        )

        self._set_objective()

    def on_validation_epoch_end(self):
        """"""
        if not self.trainer.sanity_checking:
            all_logits = torch.cat(
                [pred["logits"] for pred in self.validation_step_outputs]
            )
            all_probs = F.softmax(all_logits, dim=-1)
            labels_probs = all_probs.numpy()
            self.log_eval_report(labels_probs)

        self.valid_scorer.reset()
        self.validation_step_outputs.clear()

    def _sanity_checks(self, training_config: DictConfig) -> None:
        """
        Args:
            training_config (DictConfig):
                training configuration
        """
        super()._sanity_checks(training_config)

        if training_config.get("scorer_type") == "loose":
            assert "loose_classes" in training_config.keys(), (
                "To use a 'LooseScorer' you need to set a 'loose_classes' parameter in"
                " your training config."
            )

    def _set_objective(self) -> None:
        """"""
        objective = self.config.get("objective", "ce")
        self.smoothing = self.config.get("smoothing", 0.0)
        self.class_weights = self.config.get("class_weights", [1.0] * self.num_labels)

        if objective == "lsl" and self.smoothing == 0.0:
            logging.warning(
                "You are using label smoothing and the smoothing parameteris set to 0.0."
            )
        elif objective == "weighted" and all([w == 1.0 for w in self.class_weights]):
            logging.warning(
                "You are using a weighted CrossEntropy but the class"
                "weights are all equal to 1.0."
            )
        self.objective = {
            "ce": CrossEntropyLoss(),
            "lsl": LabelSmoothingLoss(
                nb_classes=self.num_labels, smoothing=self.smoothing
            ),
            "weighted": CrossEntropyLoss(weight=torch.Tensor(self.class_weights)),
        }[objective]

    def _set_scorers(self, scorer: Optional[Scorer]) -> None:
        """
        Method to set the scorers to use to evaluate the model.

        Args:
            scorer (Optional[Scorer]):
                helper object to compute performance metrics during training
        """
        if scorer is None:
            scorer = BaseSequenceClassificationScorer(self.num_labels)

        self.scorer = deepcopy(scorer)
        self.valid_scorer = deepcopy(scorer)
        self.test_scorer = deepcopy(scorer)

    def loss(self, labels: torch.Tensor, logits: torch.Tensor, *args, **kwargs) -> Loss:
        """
        Method called for loss computation

        Args:
            logits (torch.Tensor):
                predicted logits
            labels (torch.Tensor):
                ground truth labels
        """
        return self.objective(logits.view(-1, self.num_labels), labels.view(-1).long())

    def log_eval_report(self, probs: np.array) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses, metrics as well as
        the probability distribution of all labels.

        Args:
            probs (np.array):
                predicted probabilities
        """
        super().log_eval_report()

        for i in range(probs.shape[1]):
            fig = plt.figure(figsize=(15, 15))
            sns.histplot(probs[:, i], bins=100)
            plt.title("Probability boxplot for label {}".format(i))
            self.logger.experiment.add_figure("eval/dist_label_{}".format(i), fig)
            plt.close("all")


class BaseSeq2SeqTransformerModule(BaseTransformerModule):
    """
    Base class to extend for Transformer based seq2seq tasks.

    Args:
        training_config (DictConfig):
            training configuration
        pretrained_model (str):
            name of the pretrained Transformer model to use
        task (str):
            name of the sequence to sequence task to perform
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        scorer (Scorer):
            helper object to compute performance metrics during training
    """

    BASE_CLASS_MODEL = AutoModelForSeq2SeqLM

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        task: str,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        super().__init__(training_config, pretrained_model, model, scorer, **kwargs)
        self._sanity_checks(training_config)
        self.task = task

        self._set_objective()

    def on_validation_epoch_end(self):
        """"""
        if not self.trainer.sanity_checking:
            self.log_eval_report()

        self.valid_scorer.reset()
        self.validation_step_outputs.clear()

    def _set_objective(self) -> None:
        """"""
        self.objective = CrossEntropyLoss(ignore_index=-100)

    def _set_scorers(self, scorer: Optional[Scorer]) -> None:
        """
        Method to set the scorers to use to evaluate the model.
        """
        if scorer is None:
            scorer = LMScorer(tokenizer_name=self.pretrained_model, do_mismatch=False)

        self.scorer = scorer
        self.valid_scorer = deepcopy(scorer)
        self.test_scorer = deepcopy(scorer)

    def loss(self, labels: torch.Tensor, logits: torch.Tensor, *args, **kwargs) -> Loss:
        """
        Method called for loss computation

        Args:
            logits (torch.Tensor):
                predicted logits
            labels (torch.Tensor):
                ground truth labels

        Returns:
            Loss: computed loss value
        """
        return self.objective(logits.view(-1, logits.size(-1)), labels.view(-1))

    def log_eval_report(self) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses
        """
        super().log_eval_report()
