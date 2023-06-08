import logging

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from ..utils.losses import LabelSmoothingLoss
from ..utils.scorers import Scorer


class BowLogisticRegression(pl.LightningModule):
    """
    Logistic Regression leveraging bag of n-grams representation of a text.

    Args:
        vocab_size (int):
            size of the vocabulary used for the bag of n-grams representation
        num_labels (int):
            number of labels for the classification task
        training_config (DictConfig):
            training configuration to use
    """

    def __init__(
        self, vocab_size: int, num_labels: int, training_config: DictConfig, **kwargs
    ):
        super().__init__()
        self._sanity_checks(training_config)
        self.config = training_config
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []

        self._build_model()
        self._set_objective()
        self._set_scorers()

    def forward(self, features: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):
                Features tensor to feed to the Logistic Regression
        Returns:
            torch.Tensor: model's predictions
        """
        return self.classifier(features.float())

    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        logits = self.forward(**batch)
        loss = self.loss(logits, batch["labels"])

        self.scorer.add(logits, batch["labels"], loss.detach().cpu())

        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {
                key: torch.stack(val).mean() for key, val in self.scorer.losses.items()
            }
            self.log_dict({f"train/loss_{key}": val for key, val in logging_loss.items()})
            self.log("train/acc", self.scorer.acc)
            self.scorer.reset()
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        logits = self.forward(**batch)
        loss = self.loss(logits, batch["labels"])

        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append({"loss": loss, "logits": logits.cpu()})
        return loss

    def on_validation_epoch_end(self):
        """"""
        all_logits = torch.cat([pred["logits"] for pred in self.validation_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()

    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append({"loss": loss, "logits": logits.cpu()})
        return loss

    def configure_optimizers(self) -> Adam:
        """
        Method to define optimizer to use for training

        Returns:
            Adam: optimizer used for training
        """
        return Adam(self.parameters(), lr=self.config.learning_rates[0])

    def loss(
        self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the loss for the current batch.

        Args:
            logits (torch.Tensor):
                predicted logits
            labels (torch.Tensor):
                ground truth labels
        Returns:
            torch.Tensor: batch loss
        """
        return self.objective(logits.view(-1, self.num_labels), labels.view(-1))

    def _build_model(self) -> None:
        """"""
        self.classifier = torch.nn.Linear(self.vocab_size, self.num_labels)

    @staticmethod
    def _sanity_checks(training_config) -> None:
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

    def _set_scorers(self) -> None:
        """
        Method to set the scorers to use to evaluate the model.
        """
        self.scorer = Scorer(self.num_labels)
        self.valid_scorer = Scorer(self.num_labels)
        self.test_scorer = Scorer(self.num_labels)

    def _set_objective(self) -> None:
        """
        Method defining the loss to optimize during training.
        """
        objective = self.config.get("objective", "ce")
        self.smoothing = self.config.get("smoothing", 0.0)
        self.class_weights = self.config.get("class_weights", [1.0] * self.num_labels)

        if objective == "lsl" and self.smoothing == 0.0:
            logging.warning(
                "You are using label smoothing and the smoothing parameter"
                "is set to 0.0."
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

    def log_eval_report(self, probs: np.array) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses, metrics as well as
        the probability distribution of all labels.

        Args:
            probs (np.array):
                predicted probabilities by the model
        """
        table = self.valid_scorer.get_table()
        self.logger.experiment.add_text("eval/report", table)

        logging_loss = {
            key: torch.stack(val).mean() for key, val in self.valid_scorer.losses.items()
        }
        self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})

        eval_report = self.valid_scorer.to_dict()
        for key, value in eval_report.items():
            if not isinstance(value, list) and not isinstance(value, np.ndarray):
                self.log("eval/{}".format(key), value)

        for i in range(probs.shape[1]):
            fig = plt.figure(figsize=(15, 15))
            sns.distplot(probs[:, i], kde=False, bins=100)
            plt.title("Probability boxplot for label {}".format(i))
            self.logger.experiment.add_figure("eval/dist_label_{}".format(i), fig)
            plt.close("all")
