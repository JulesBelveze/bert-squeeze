from __future__ import annotations

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
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from ..utils.experiment_logging import ExperimentLogger
from ..utils.losses import LabelSmoothingLoss
from ..utils.optimizers import BertAdam
from ..utils.scorers import BaseSequenceClassificationScorer, LMScorer, Scorer
from ..utils.types import (
    FastBertLoss,
    SequenceClassificationOutput,
    SequenceClassificationStepOutput,
)


class _IdentityParamList(list):
    def __contains__(self, item: object) -> bool:
        return any(param is item for param in self)


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
        if model is None:
            if self.BASE_CLASS_MODEL is None:
                raise TypeError("A base model class or model instance is required.")
            model = self.BASE_CLASS_MODEL.from_pretrained(self.pretrained_model)
        self.model = model
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []

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
        learning_rate = (
            self.config.learning_rates[0]
            if isinstance(self.config.learning_rates, ListConfig)
            else self.config.learning_rate
        )

        optimizer_name = self.config.get("optimizer", "adamw")
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=learning_rate,
                eps=self.config.adam_eps,
            )

            if self.config.lr_scheduler:
                scheduler = ReduceLROnPlateau(optimizer)
                lr_scheduler = {'scheduler': scheduler, 'name': 'NeptuneLogger'}
                return [optimizer], [lr_scheduler]

        elif optimizer_name == "bertadam":
            optimizer = BertAdam(
                optimizer_parameters,
                lr=learning_rate,
                warmup=self.config.warmup_ratio,
            )

        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=learning_rate,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_parameters,
                lr=learning_rate,
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
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight', 'layer_norm.weight']

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
                        'weight_decay': self.config.weight_decay,
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
                        'weight_decay': 0.0,
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
                    'weight_decay': self.config.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay': 0.0,
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
                    'weight_decay': self.config.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    'weight_decay': 0.0,
                },
            ]
        for group in optimizer_grouped_parameters:
            group["params"] = _IdentityParamList(list(group["params"]))
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

    def loss(self, *args, **kwargs) -> object:
        """"""
        raise NotImplementedError()

    def log_eval_report(self, *args) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses and metrics
        """
        eval_report = self.valid_scorer.to_dict()
        try:
            table = self.valid_scorer.get_table(eval_report)
        except TypeError:
            table = self.valid_scorer.get_table()
        exp_logger = ExperimentLogger.from_module(self)
        exp_logger.add_text("eval/report", table)

        logging_loss = {}
        for key, values in self.valid_scorer.losses.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, torch.Tensor):
                logging_loss[key] = torch.stack(values).mean()
            else:
                logging_loss[key] = torch.tensor(np.mean(values))

        if logging_loss:
            self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})

        for metric, value in eval_report.items():
            if isinstance(value, dict):
                self.log_dict({f"eval/{metric}/{key}": v for key, v in value.items()})
            elif not isinstance(value, (list, np.ndarray)):
                self.log(f"eval/{metric}", value)


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
        self.num_labels = num_labels
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model, num_labels=num_labels
        )

        if model is None:
            model = self.BASE_CLASS_MODEL.from_pretrained(
                pretrained_model, config=self.model_config
            )

        super().__init__(training_config, pretrained_model, model, scorer, **kwargs)
        self._sanity_checks(training_config)
        self._set_objective()

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        self._before_training_step()
        step_output = self._classification_step(batch, training=True)
        self._update_scorer(self.scorer, step_output)
        self._log_training_metrics()
        return step_output.optimization_loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        step_output = self._classification_step(batch)
        self._update_scorer(self.valid_scorer, step_output)
        self._store_step_output(self.validation_step_outputs, step_output)
        return step_output.optimization_loss

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        step_output = self._classification_step(batch)
        self._update_scorer(self.test_scorer, step_output)
        self._store_step_output(self.test_step_outputs, step_output)
        return step_output.optimization_loss

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        self._before_prediction_step()
        return self._predict_probabilities(batch)

    def _classification_step(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool = False,
    ) -> SequenceClassificationStepOutput:
        output = self._classification_output(batch)
        labels = batch["labels"]
        loss = self._classification_loss(output, labels)
        return SequenceClassificationStepOutput(output=output, labels=labels, loss=loss)

    def _classification_output(
        self, batch: Dict[str, torch.Tensor]
    ) -> SequenceClassificationOutput:
        logits = self.forward(**self._model_inputs(batch))
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Sequence classifiers must return tensor logits.")
        return SequenceClassificationOutput(logits=logits)

    def _classification_loss(
        self,
        output: SequenceClassificationOutput,
        labels: torch.Tensor,
    ) -> Union[torch.Tensor, FastBertLoss]:
        return self.loss(labels=labels, logits=output.logits)

    def _model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_names = ("input_ids", "attention_mask", "token_type_ids")
        return {name: batch[name] for name in input_names if name in batch}

    def _before_training_step(self) -> None:
        pass

    def _before_prediction_step(self) -> None:
        pass

    def _predict_probabilities(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self._classification_output(batch)
        return F.softmax(output.logits, dim=-1)

    def _log_training_metrics(self) -> None:
        if self.global_step <= 0 or self.global_step % self.config.logging_steps != 0:
            return

        logging_loss = {
            key: torch.stack(values).mean()
            for key, values in self.scorer.losses.items()
            if values
        }
        self.log_dict({f"train/loss_{key}": value for key, value in logging_loss.items()})
        self.log("train/acc", self.scorer.acc)
        self.scorer.reset()

    @staticmethod
    def _detached_logits(
        logits: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(logits, torch.Tensor):
            return logits.detach().cpu()
        return [item.detach().cpu() for item in logits]

    def _update_scorer(
        self,
        scorer: Scorer,
        step_output: SequenceClassificationStepOutput,
    ) -> None:
        logits = self._detached_logits(step_output.output.scorer_logits)
        labels = step_output.labels.detach().cpu()
        loss = step_output.loss
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu()
        scorer.add(logits, labels, loss)

    @staticmethod
    def _store_step_output(
        output_store: List[Dict[str, torch.Tensor]],
        step_output: SequenceClassificationStepOutput,
    ) -> None:
        output_store.append(
            {
                "loss": step_output.optimization_loss.detach().cpu(),
                "logits": step_output.output.logits.detach().cpu(),
                "labels": step_output.labels.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self):
        """"""
        if not self.trainer.sanity_checking and self.validation_step_outputs:
            all_logits = torch.cat(
                [pred["logits"] for pred in self.validation_step_outputs]
            )
            all_probs = F.softmax(all_logits, dim=-1)
            labels_probs = all_probs.numpy()
            self.log_eval_report(labels_probs)

        self.valid_scorer.reset()
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        logging.info(self.test_scorer.get_table())
        self.test_scorer.reset()
        self.test_step_outputs.clear()

    @staticmethod
    def _sanity_checks(training_config: DictConfig) -> None:
        """
        Args:
            training_config (DictConfig):
                training configuration
        """
        BaseTransformerModule._sanity_checks(training_config)

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
        elif objective == "weighted" and all(w == 1.0 for w in self.class_weights):
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
            scorer = BaseSequenceClassificationScorer(list(range(self.num_labels)))

        self.scorer = deepcopy(scorer)
        self.valid_scorer = deepcopy(scorer)
        self.test_scorer = deepcopy(scorer)

    def loss(
        self, labels: torch.Tensor, logits: torch.Tensor, *args, **kwargs
    ) -> Union[torch.Tensor, FastBertLoss]:
        """
        Method called for loss computation

        Args:
            logits (torch.Tensor):
                predicted logits
            labels (torch.Tensor):
                ground truth labels
        """
        return self.objective(logits.view(-1, self.num_labels), labels.view(-1).long())

    def log_eval_report(self, probs: np.ndarray) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses, metrics as well as
        the probability distribution of all labels.

        Args:
            probs (np.array):
                predicted probabilities
        """
        super().log_eval_report()

        exp_logger = ExperimentLogger.from_module(self)
        for i in range(probs.shape[1]):
            fig = plt.figure(figsize=(15, 15))
            sns.histplot(probs[:, i], bins=100)
            plt.title(f"Probability boxplot for label {i}")
            exp_logger.add_figure(f"eval/dist_label_{i}", fig)
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

    def loss(
        self, labels: torch.Tensor, logits: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
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
