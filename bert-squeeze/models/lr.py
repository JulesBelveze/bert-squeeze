import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from ..utils.losses import LabelSmoothingLoss
from ..utils.scorer import Scorer


class BowLogisticRegression(pl.LightningModule):
    def __init__(self, vocab_size: int, num_labels: int, training_config: DictConfig, **kwargs):
        super().__init__()
        self._sanity_check(training_config)
        self.config = training_config
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        self._build_model()
        self._set_objective()
        self._set_scorers()

    @staticmethod
    def _sanity_check(training_config):
        assert training_config.logging_steps > 0, \
            "'logging_steps' should be strictly greater than 0"
        assert training_config.accumulation_steps > training_config.logging_steps, \
            "'logging_steps' should be greater than 'accumulation_steps'"

    def _set_scorers(self):
        self.scorer = Scorer(self.num_labels)
        self.valid_scorer = Scorer(self.num_labels)
        self.test_scorer = Scorer(self.num_labels)

    def _set_objective(self):
        objective = self.config.get("objective", "ce")
        self.smoothing = self.config.get("smoothing", 0.0)
        self.class_weights = self.config.get("class_weights", [1.0] * self.num_labels)

        if objective == "lsl" and self.smoothing == 0.0:
            logging.warning("You are using label smoothing and the smoothing parameter"
                            "is set to 0.0.")
        elif objective == "weighted" and all([w == 1.0 for w in self.class_weights]):
            logging.warning("You are using a weighted CrossEntropy but the class"
                            "weights are all equal to 1.0.")
        self.objective = {
            "ce": CrossEntropyLoss(),
            "lsl": LabelSmoothingLoss(classes=self.num_labels,
                                      smoothing=self.smoothing),
            "weighted": CrossEntropyLoss(weight=torch.Tensor(self.class_weights)),
        }[objective]

    def _build_model(self):
        self.classifier = torch.nn.Linear(self.vocab_size, self.num_labels)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.learning_rates[0])

    def forward(self, features: torch.Tensor = None, **kwargs):
        return self.classifier(features.float())

    def loss(self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        return self.objective(logits.view(-1, self.num_labels), labels.view(-1))

    def shared_step(self, batch):
        logits = self.forward(**batch)
        loss = self.loss(logits, batch["labels"])
        return loss, logits

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, logits = self.shared_step(batch)
        self.scorer.add(logits, batch["labels"], loss.detach().cpu())

        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"loss_{key}"].log(value)

            self.logger.experiment["train/loss"].log(value=logging_loss, step=self.global_step)
            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    def validation_epoch_end(self, test_step_outputs: List[dict]):
        all_logits = torch.cat([pred["logits"] for pred in test_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()

    def log_eval_report(self, probs: np.array):
        table = self.valid_scorer.get_table()
        self.logger.experiment["eval/report"].log(table)

        logging_loss = {key: torch.stack(val).mean() for key, val in self.valid_scorer.losses.items()}
        for key, value in logging_loss.items():
            self.logger.experiment[f"eval/loss_{key}"].log(value)

        eval_report = self.valid_scorer.to_dict()
        for key, value in eval_report.items():
            if not isinstance(value, list) and not isinstance(value, np.ndarray):
                self.logger.experiment["eval/{}".format(key)].log(value=value, step=self.global_step)

        for i in range(probs.shape[1]):
            fig = plt.figure(figsize=(15, 15))
            sns.distplot(probs[:, i], kde=False, bins=100)
            plt.title("Probability boxplot for label {}".format(i))
            self.logger.experiment["eval/dist_label_{}".format(i)].log(fig)
            plt.close("all")
