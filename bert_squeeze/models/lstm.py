import logging
from typing import List

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from ..utils.losses import LabelSmoothingLoss
from ..utils.scorers import Scorer


class LtLSTM(pl.LightningModule):
    """
    Lightning module to train a LSTM-based model on a sequence classification task.

    Args:
        training_config (DictConfig):
            training configuration to use
        vocab_size (int):
            size of the vocabulary used for the bag of n-grams representation
        hidden_dim (int):
            number of features in the hidden states
        num_labels (int):
            number of labels for the classification task
    """

    def __init__(
        self,
        training_config: DictConfig,
        vocab_size: int,
        hidden_dim: int,
        num_labels: int,
        *args,
        **kwargs,
    ):
        super(LtLSTM, self).__init__()
        self._sanity_checks(training_config)

        self.config = training_config
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []

        self._set_scorers()
        self._set_objective()
        self._build_model()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor):
                Tokens tensor to feed to the Logistic Regression
        Returns:
            torch.Tensor: model's predictions
        """
        bs, len_seq = tokens.shape
        embeds = self.embedding(tokens).view(len_seq, bs, -1)

        (h0, c0) = (
            Variable(torch.zeros(2, bs, self.hidden_dim)),
            Variable(torch.zeros(2, bs, self.hidden_dim)),
        )

        lstm_out, _ = self.lstm(embeds, (h0, c0))
        features = lstm_out[-1]
        features = self.drop(features)
        return self.classifier(features)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        logits = self.forward(tokens=batch["features"])
        loss = self.loss(logits, batch["labels"])

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss.detach().cpu())
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
        logits = self.forward(tokens=batch["features"])
        loss = self.loss(logits, batch["labels"])

        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """"""
        all_logits = torch.cat([pred["logits"] for pred in self.validation_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()

    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        logits = self.forward(tokens=batch["features"])
        loss = self.loss(logits, batch["labels"])

        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def on_test_epoch_end(self) -> None:
        """"""
        print(self.test_scorer.get_table())
        self.test_scorer.reset()

    def configure_optimizers(self) -> Adam:
        """
        Method to define optimizer to use for training

        Returns:
            Adam: optimizer used for training
        """
        return Adam(self.parameters(), lr=self.config.learning_rates[0])

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

    def _build_model(self) -> None:
        """"""
        self.embedding = torch.nn.Embedding(self.vocab_size, 300)
        self.lstm = torch.nn.LSTM(
            input_size=300, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True
        )
        self.drop = torch.nn.Dropout(p=self.config.dropout)
        self.classifier = torch.nn.Linear(2 * self.hidden_dim, self.num_labels)

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

    def _set_scorers(self) -> None:
        """
        Method to set the scorers to use to evaluate the model.
        """
        self.scorer = Scorer(self.num_labels)
        self.valid_scorer = Scorer(self.num_labels)
        self.test_scorer = Scorer(self.num_labels)

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

    def log_eval_report(self, probs: np.array) -> None:
        """
        Method that logs an evaluation report.

        It uses the evaluation scorer to log all the available losses, metrics as well as
        the probability distribution of all labels.

        Args:
            probs (np.array):
                predicted probabilities
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
