import math
from typing import Dict, List, Tuple

import logging
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, ListConfig
from pkg_resources import resource_filename
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from ..utils.losses import LabelSmoothingLoss
from ..utils.losses.distillation_losses import KLDivLoss
from ..utils.optimizers import BertAdam
from ..utils.scorer import Scorer
from ..utils.types import DistillationLoss


class Distiller(pl.LightningModule):
    def __init__(self, teacher_config: DictConfig, student_config: DictConfig, training_config: DictConfig,
                 **kwargs):
        super().__init__()
        self.params = training_config
        self.teacher_config = teacher_config
        self.student_config = student_config
        self.config = training_config

        self.teacher = self.build_teacher(teacher_config=teacher_config)
        self.student = instantiate(student_config)

        self._set_objectives()
        self._set_scorers()

    def _set_objectives(self) -> None:
        """"""
        objective = self.params.get("objective", "ce")
        distillation_loss = self.params.get("distillation_loss", "mse")

        self.smoothing = self.params.get("smoothing", 0.0)
        self.class_weights = self.params.get("class_weights", [1.0] * self.params.num_labels)

        if objective == "lsl" and self.params.smoothing == 0.0:
            logging.warning("You are using label smoothing and the smoothing parameter"
                            "is set to 0.0.")
        elif objective == "weighted" and all([w == 1.0 for w in self.params.get("class_weights", None)]):
            logging.warning("You are using a weighted CrossEntropy but the class"
                            "weights are all equal to 1.0.")
        self.l_lce = {
            "ce": CrossEntropyLoss(),
            "lsl": LabelSmoothingLoss(classes=self.params.num_labels, smoothing=self.params.smoothing),
            "weighted": CrossEntropyLoss(
                weight=torch.Tensor(self.params.class_weights) if self.params.get("class_weights") is not None
                else None
            ),
        }[objective]

        # distillation loss
        self.l_distill = {
            "mse": torch.nn.MSELoss(),
            "kl": KLDivLoss()
        }[distillation_loss]

    def _set_scorers(self) -> None:
        """"""
        self.s_scorer = Scorer(self.params.num_labels)
        self.s_valid_scorer = Scorer(self.params.num_labels)
        self.s_test_scorer = Scorer(self.params.num_labels)

    def _get_student_parameters(self):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']

        if self.params.discriminative_learning:
            if isinstance(self.params.learning_rates, ListConfig) and len(self.params.learning_rates) > 1:
                groups = [(f'layer.{i}.', self.params.learning_rates[i]) for i in range(12)]
            else:
                lr = self.params.learning_rates[0] if isinstance(self.params.learning_rates,
                                                                 ListConfig) else self.params.learning_rates
                groups = [(f'layer.{i}.', lr * pow(self.params.layer_lr_decay, 11 - i)) for i in range(12)]

            group_all = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters, decay_optimizer_parameters = [], []
            for g, l in groups:
                no_decay_optimizer_parameters.append(
                    {'params': [p for n, p in self.student.named_parameters() if
                                not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                     'weight_decay_rate': self.params.weight_decay, 'lr': l}
                )
                decay_optimizer_parameters.append(
                    {'params': [p for n, p in self.student.named_parameters() if
                                any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                     'weight_decay_rate': 0.0, 'lr': l}
                )

            group_all_parameters = [
                {'params': [p for n, p in self.student.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
                 'weight_decay_rate': self.params.weight_decay},
                {'params': [p for n, p in self.student.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
                 'weight_decay_rate': 0.0},
            ]
            optimizer_grouped_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters \
                                           + group_all_parameters
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.student.named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.params.weight_decay},
                {'params': [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        return optimizer_grouped_parameters

    def configure_optimizers(self) -> Tuple[List, List]:
        """"""
        optimizer_parameters = self._get_student_parameters()
        if self.student_config.architecture == "lr":
            optimizer = torch.optim.SGD(optimizer_parameters, lr=self.params.learning_rates[0])
            return [optimizer], []

        elif self.student_config.architecture == "transformer":
            if self.params.optimizer == "adamw":
                optimizer = AdamW(optimizer_parameters, lr=self.params.learning_rates[0],
                                  eps=self.params.adam_eps)

                if self.params.lr_scheduler:
                    num_training_steps = len(self.train_dataloader()) * self.params.num_epochs // \
                                         self.params.accumulation_steps

                    warmup_steps = math.ceil(num_training_steps * self.params.warmup_ratio)
                    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=warmup_steps,
                                                                num_training_steps=num_training_steps)
                    lr_scheduler = {"scheduler": scheduler, "name": "NeptuneLogger"}
                    return [optimizer], [lr_scheduler]
            elif self.params.optimizer == "bertadam":
                num_training_steps = len(self.train_dataloader()) * self.params.num_epochs // \
                                     self.params.accumulation_steps
                optimizer = BertAdam(optimizer_parameters, lr=self.params.learning_rates[0],
                                     warmup=self.params.warmup_ratio, t_total=num_training_steps)
            elif self.params.optimizer == "adam":
                optimizer = torch.optim.Adam(optimizer_parameters, lr=self.params.learning_rates[0])
            else:
                raise ValueError(f"Optimizer '{self.params.optimizer}' not supported.")

            return [optimizer], []
        else:
            raise ValueError(f"Student model '{self.student_config.architecture}' not supported.")

    def build_teacher(self, teacher_config: DictConfig):
        """"""
        checkpoint_path = resource_filename("bert-squeeze", teacher_config.checkpoint_path)

        teacher_class = get_class(teacher_config._target_)
        teacher = teacher_class.load_from_checkpoint(
            checkpoint_path,
            training_config=teacher_config,
            pretrained_model=teacher_config.pretrained_model,
            num_labels=self.params.num_labels
        )
        logging.info("Teacher successfully loaded.")
        return teacher

    def get_teacher_logits(self, batch) -> torch.Tensor:
        """"""
        self.teacher.eval()
        teacher_inputs = {key[2:]: val for key, val in batch.items() if key.startswith("t_")}
        with torch.no_grad():
            logits = self.teacher.forward(**teacher_inputs)
        return logits

    def get_student_logits(self, batch) -> torch.Tensor:
        """"""
        student_inputs = {key[2:]: val for key, val in batch.items() if key.startswith("s_")}
        logits = self.student.forward(**student_inputs)
        return logits

    def loss(self, teacher_logits, student_logits, labels, ignore_index: int = -100) -> DistillationLoss:
        """Return loss for knowledge distillation"""
        # Ignore soft labeled indices (where label is -100)
        active_idx = (labels != ignore_index)
        if active_idx.sum().item() > 0:
            objective = self.l_lce(student_logits[active_idx], labels[active_idx])
        else:
            objective = torch.tensor(0.0).to(labels.device)

        kd_loss = self.l_distill(teacher_logits, student_logits)
        full_loss = (1 - self.params.alpha) * kd_loss + self.params.alpha * objective
        return DistillationLoss(
            kd_loss=kd_loss,
            objective=objective,
            full_loss=full_loss
        )

    def training_step(self, batch, _) -> torch.Tensor:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits, batch["s_labels"])

        self.s_scorer.add(s_logits.detach().cpu(), batch["s_labels"].cpu(), loss)
        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.s_scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"loss_{key}"].log(value)

            self.logger.experiment["train/acc"].log(self.s_scorer.acc, step=self.global_step)
        return loss.full_loss

    def test_step(self, batch, _) -> Dict:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits, batch["s_labels"])
        self.s_test_scorer.add(s_logits.detach().cpu(), batch["labels"].detach().cpu(), loss)
        return {"loss": loss.full_loss, "logits": s_logits.detach().cpu()}

    def validation_step(self, batch, _) -> Dict:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits, batch["s_labels"])
        self.s_valid_scorer.add(s_logits.detach().cpu(), batch["s_labels"].detach().cpu(), loss)
        return {"loss": loss.full_loss, "logits": s_logits.detach().cpu()}

    def training_epoch_end(self, _) -> None:
        """"""
        self.s_scorer.reset()

    def validation_epoch_end(self, test_step_outputs: List[Dict]) -> None:
        """"""
        all_logits = torch.cat([pred["logits"] for pred in test_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = [all_probs[:, i] for i in range(all_probs.shape[-1])]

        self.log_eval_report(labels_probs)
        self.s_valid_scorer.reset()

    def test_epoch_end(self, test_step_outputs: List[Dict]) -> None:
        """"""
        all_logits = torch.cat([pred["logits"] for pred in test_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = [all_probs[:, i] for i in range(all_probs.shape[-1])]

        self.log_eval_report(labels_probs)
        self.s_test_scorer.reset()

    def log_eval_report(self, probs: List[np.array]) -> None:
        """"""
        table = self.s_valid_scorer.get_table()
        self.logger.experiment["eval/report"].log(table)

        # logging losses to neptune
        logging_loss = {key: torch.stack(val).mean() for key, val in self.s_valid_scorer.losses.items()}
        for key, value in logging_loss.items():
            self.logger.experiment[f"eval/loss_{key}"].log(value)

        # logging other metrics
        eval_report = self.s_valid_scorer.to_dict()
        for key, value in eval_report.items():
            if not isinstance(value, list) and not isinstance(value, np.ndarray):
                self.logger.experiment["eval/{}".format(key)].log(value=value, step=self.global_step)

        # logging probability distributions
        for i in range(len(probs)):
            fig = plt.figure(figsize=(15, 15))
            sns.distplot(probs[i], kde=False, bins=100)
            plt.title("Probability boxplot for label {}".format(i))
            self.logger.experiment["eval/dist_label_{}".format(i)].log(fig)
            plt.close("all")
