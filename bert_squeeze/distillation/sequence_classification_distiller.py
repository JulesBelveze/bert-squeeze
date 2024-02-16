import logging
from typing import Any, Dict, List, Tuple, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from overrides import overrides
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

from bert_squeeze.distillation.base_distiller import BaseDistiller
from bert_squeeze.utils.losses import LabelSmoothingLoss
from bert_squeeze.utils.losses.distillation_losses import KLDivLoss
from bert_squeeze.utils.scorers import BaseSequenceClassificationScorer
from bert_squeeze.utils.types import DistillationLoss


class BaseSequenceClassificationDistiller(BaseDistiller):
    """
    Lightning module to distil a given teacher model into a given student one for sequence classification task.

    Args:
        teacher (Union["pl.LightningModule", "torch.nn.Module"]):
            model to distil knowledge from
        student (Union["pl.LightningModule", "torch.nn.Module"]):
            model to use as a student
        training_config (DictConfig):
            configuration to use for training and to distil the teacher model
        teacher_checkpoint (str):
            path to checkpoints to load to the teacher model
        labels (Union[List[str], List[int]]):
            list of labels to use for sequence classification
    """

    def __init__(
        self,
        teacher: Union["pl.LightningModule", "torch.nn.Module"],
        student: Union["pl.LightningModule", "torch.nn.Module"],
        training_config: DictConfig,
        labels: Union[List[str], List[int]],
        teacher_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__(teacher, student, training_config, teacher_checkpoint, **kwargs)
        self.labels = labels

        self._set_objectives()
        self._set_scorers()

    def _set_objectives(self) -> None:
        """
        Sets the different objectives used for distillation:
        - a classical one to evaluate the student's predictions
        - a distillation loss to evaluate the closeness of the student's predictions to the
          teacher's ones.
        """
        objective = self.params.get("objective", "ce")
        distillation_loss = self.params.get("distillation_loss", "mse")

        self.smoothing = self.params.get("smoothing", 0.0)
        self.class_weights = self.params.get(
            "class_weights", [1.0] * self.params.num_labels
        )

        if objective == "lsl" and self.params.smoothing == 0.0:
            logging.warning(
                "You are using label smoothing and the smoothing parameteris set to 0.0."
            )
        elif objective == "weighted" and all(
            [w == 1.0 for w in self.params.get("class_weights", None)]
        ):
            logging.warning(
                "You are using a weighted CrossEntropy but the class"
                "weights are all equal to 1.0."
            )
        self.loss_ce = {
            "ce": CrossEntropyLoss(),
            "lsl": LabelSmoothingLoss(
                nb_classes=self.params.num_labels, smoothing=self.params.smoothing
            ),
            "weighted": CrossEntropyLoss(
                weight=(
                    torch.Tensor(self.params.class_weights)
                    if self.params.get("class_weights") is not None
                    else None
                )
            ),
        }[objective]

        self.loss_distill = {"mse": torch.nn.MSELoss(), "kl": KLDivLoss()}[
            distillation_loss
        ]

    def _set_scorers(self) -> None:
        """
        Method to set the scorers to use to evaluate the model.
        """
        self.s_scorer = BaseSequenceClassificationScorer(self.labels)
        self.s_valid_scorer = BaseSequenceClassificationScorer(self.labels)
        self.s_test_scorer = BaseSequenceClassificationScorer(self.labels)

    def get_teacher_logits(self, batch: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError()

    def get_student_logits(self, batch: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError()

    def log_eval_report(self, probs: List[np.array]) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses, metrics as well as
        the probability distribution of all labels.

        Args:
            probs (List[np.array]):
                predicted probabilities
        """
        super().log_eval_report()

        # logging probability distributions
        for i in range(len(probs)):
            fig = plt.figure(figsize=(15, 15))
            sns.distplot(probs[i], kde=False, bins=100)
            plt.title("Probability boxplot for label {}".format(i))
            self.logger.experiment.add_figure("eval/dist_label_{}".format(i), fig)
            plt.close("all")


class SequenceClassificationDistiller(BaseSequenceClassificationDistiller):
    """
    Lightning module to distil a given teacher model into a given student one for sequence classification task.

    Args:
        teacher (Union["pl.LightningModule", "torch.nn.Module"]):
            model to distil knowledge from
        student (Union["pl.LightningModule", "torch.nn.Module"]):
            model to use as a student
        training_config (DictConfig):
            configuration to use for training and to distil the teacher model
        teacher_checkpoint (str):
            path to checkpoints to load to the teacher model
        labels (Union[List[str], List[int]]):
            list of labels to use for sequence classification
    """

    def __init__(
        self,
        teacher: Union["pl.LightningModule", "torch.nn.Module"],
        student: Union["pl.LightningModule", "torch.nn.Module"],
        training_config: DictConfig,
        labels: Union[List[str], List[int]],
        teacher_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__(
            teacher, student, training_config, labels, teacher_checkpoint, **kwargs
        )

    @overrides
    def get_teacher_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get teacher's predictions.

        Args:
            batch (Dict[str, torch.Tensor]):
                batched features
        Returns:
            torch.Tensor:
                teacher logits
        """
        self.teacher.eval()
        teacher_inputs = {
            key[2:]: val
            for key, val in batch.items()
            if key.startswith("t_") and "labels" not in key
        }
        with torch.no_grad():
            outputs = self.teacher.forward(**teacher_inputs)

        if isinstance(outputs, SequenceClassifierOutput):
            return outputs.logits

        return outputs

    @overrides
    def get_student_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get student's predictions.

        Args:
            batch (Dict[str, torch.Tensor]):
                batched features
        Returns:
            torch.Tensor:
                student logits
        """
        student_inputs = {
            key[2:]: val
            for key, val in batch.items()
            if key.startswith("s_") and "labels" not in key
        }
        outputs = self.student.forward(**student_inputs)

        if isinstance(outputs, SequenceClassifierOutput):
            return outputs.logits

        return outputs

    @overrides
    def loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor = None,
        ignore_index: int = -100,
        *args,
        **kwargs,
    ) -> DistillationLoss:
        """
        Method called for loss computation

        Args:
            teacher_logits (torch.Tensor):
                teacher's predictions
            student_logits (torch.Tensor):
                student's predictions
            labels (torch.Tensor):
                ground truth labels
            ignore_index (int):
                labels to ignore during loss computation
        Returns:

        """
        # Ignore soft labeled indices (where label is `ignore_index`)
        active_idx = labels != ignore_index
        if active_idx.sum().item() > 0:
            objective = self.loss_ce(student_logits[active_idx], labels[active_idx])
        else:
            objective = torch.tensor(0.0).to(labels.device)

        kd_loss = self.loss_distill(teacher_logits, student_logits)
        full_loss = (1 - self.params.alpha) * kd_loss + self.params.alpha * objective
        return DistillationLoss(kd_loss=kd_loss, objective=objective, full_loss=full_loss)

    @overrides
    def training_step(self, batch, _) -> torch.Tensor:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits, batch["s_labels"])

        self.s_scorer.add(s_logits.detach().cpu(), batch["s_labels"].cpu(), loss)
        if self.global_step > 0 and self.global_step % self.params.logging_steps == 0:
            logging_loss = {
                f"train/{key}": torch.stack(val).mean()
                for key, val in self.s_scorer.losses.items()
            }
            self.log_dict(logging_loss)

            self.log("train/acc", self.scorer.acc)
        return loss.full_loss

    @overrides
    def test_step(self, batch, _) -> Dict:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits, batch["s_labels"])
        self.s_test_scorer.add(
            s_logits.detach().cpu(), batch["labels"].detach().cpu(), loss
        )
        self.test_step_outputs.append(
            {"loss": loss.full_loss, "logits": s_logits.detach().cpu()}
        )
        return {"loss": loss.full_loss}

    @overrides
    def validation_step(self, batch, _) -> Dict:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits, batch["s_labels"])
        self.s_valid_scorer.add(
            s_logits.detach().cpu(), batch["s_labels"].detach().cpu(), loss
        )
        self.validation_step_outputs.append(
            {"loss": loss.full_loss, "logits": s_logits.detach().cpu()}
        )
        return {"loss": loss.full_loss}

    @overrides
    def on_validation_epoch_end(self) -> None:
        """"""
        if not self.trainer.sanity_checking:
            all_logits = torch.cat(
                [pred["logits"] for pred in self.validation_step_outputs]
            )
            all_probs = F.softmax(all_logits, dim=-1)
            labels_probs = [all_probs[:, i] for i in range(all_probs.shape[-1])]
            self.log_eval_report(labels_probs)

        self.s_valid_scorer.reset()

    @overrides
    def on_test_epoch_end(self) -> None:
        """"""
        all_logits = torch.cat([pred["logits"] for pred in self.test_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = [all_probs[:, i] for i in range(all_probs.shape[-1])]

        self.log_eval_report(labels_probs)
        self.s_test_scorer.reset()


class SequenceClassificationParallelDistiller(BaseSequenceClassificationDistiller):
    """
    Lightning module to distil a given teacher model into a given student one for sequence classification
    tasks.

    The main goal of such class is to distil the teacher's multilingual ability to the student.

    Args:
        teacher (Union["pl.LightningModule", "torch.nn.Module"]):
            model to distil knowledge from
        student (Union["pl.LightningModule", "torch.nn.Module"]):
            model to use as a student
        training_config (DictConfig):
            configuration to use for training and to distil the teacher model
        teacher_checkpoint (str):
            path to checkpoints to load to the teacher model
        labels (Union[List[str], List[int]]):
            list of labels to use for sequence classification
    """

    def __init__(
        self,
        teacher: Union["pl.LightningModule", "torch.nn.Module"],
        student: Union["pl.LightningModule", "torch.nn.Module"],
        training_config: DictConfig,
        labels: Union[List[str], List[int]],
        teacher_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__(
            teacher, student, training_config, labels, teacher_checkpoint, **kwargs
        )

    @overrides
    def get_teacher_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get teacher's predictions.

        Args:
            batch (Dict[str, torch.Tensor]):
                batched features
        Returns:
            torch.Tensor:
                teacher logits
        """
        self.teacher.eval()
        teacher_inputs = {
            key[2:]: val
            for key, val in batch.items()
            if key.startswith("t_") and "labels" not in key
        }
        with torch.no_grad():
            outputs = self.teacher.forward(**teacher_inputs)

        if isinstance(outputs, SequenceClassifierOutput):
            return outputs.logits
        return outputs

    @overrides
    def get_student_logits(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get student's predictions.

        Args:
            batch (Dict[str, torch.Tensor]):
                batched features
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                student logits predicted for the parallel data, resulting in one prediction for
                the original text and one prediction for the translation.
        """
        student_inputs = {
            key[2:]: val
            for key, val in batch.items()
            if key.startswith("s_") and "labels" not in key
        }
        original_outputs = self.student.forward(
            **{
                key: val
                for key, val in student_inputs.items()
                if not key.startswith("translation")
            }
        )
        if isinstance(original_outputs, SequenceClassifierOutput):
            original_outputs = original_outputs.logits

        translation_outputs = self.student.forward(
            **{
                key: val
                for key, val in student_inputs.items()
                if key.startswith("translation")
            }
        )
        if isinstance(translation_outputs, SequenceClassifierOutput):
            translation_outputs = translation_outputs.logits

        return original_outputs, translation_outputs

    @overrides
    def loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        student_logits_translation: torch.Tensor,
        *args,
        **kwargs,
    ) -> DistillationLoss:
        """
        Method called for loss computation, it computes the error between the teacher's and
        student's predictions for the original text and for the translated one.

        Args:
            teacher_logits (torch.Tensor):
                teacher's predictions
            student_logits (torch.Tensor):
                student's predictions for the original text
            student_logits_translation (torch.Tensor):
                student's predictions for the translated text

        Returns:
            DistillationLoss
        """
        kd_loss_original = self.loss_distill(teacher_logits, student_logits)
        kd_loss_translation = self.loss_distill(
            teacher_logits, student_logits_translation
        )
        full_loss = kd_loss_translation + kd_loss_original
        return DistillationLoss(
            kd_loss=kd_loss_original, objective=kd_loss_original, full_loss=full_loss
        )

    @overrides
    def training_step(self, batch, _) -> torch.Tensor:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits_original, s_logits_translated = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits_original, s_logits_translated)
        return loss.full_loss

    @overrides
    def test_step(self, batch, _) -> Dict:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits_original, s_logits_translated = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits_original, s_logits_translated)
        self.s_test_scorer.add(
            s_logits_original.detach().cpu(), batch["labels"].detach().cpu(), loss
        )
        self.test_step_outputs.append(
            {"loss": loss.full_loss, "logits": s_logits_original.detach().cpu()}
        )
        return {"loss": loss.full_loss}

    @overrides
    def validation_step(self, batch, _) -> Dict:
        """"""
        t_logits = self.get_teacher_logits(batch)
        s_logits_original, s_logits_translated = self.get_student_logits(batch)

        loss = self.loss(t_logits, s_logits_original, s_logits_translated)
        self.s_valid_scorer.add(
            s_logits_original.detach().cpu(), batch["s_labels"].detach().cpu(), loss
        )
        self.validation_step_outputs.append(
            {"loss": loss.full_loss, "logits": s_logits_original.detach().cpu()}
        )
        return {"loss": loss.full_loss}

    @overrides
    def on_validation_epoch_end(self) -> None:
        """"""
        if not self.trainer.sanity_checking:
            all_logits = torch.cat(
                [pred["logits"] for pred in self.validation_step_outputs]
            )
            all_probs = F.softmax(all_logits, dim=-1)
            labels_probs = [all_probs[:, i] for i in range(all_probs.shape[-1])]
            self.log_eval_report(labels_probs)

        self.s_valid_scorer.reset()

    @overrides
    def on_test_epoch_end(self) -> None:
        """"""
        all_logits = torch.cat([pred["logits"] for pred in self.test_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = [all_probs[:, i] for i in range(all_probs.shape[-1])]

        self.log_eval_report(labels_probs)
        self.s_test_scorer.reset()
