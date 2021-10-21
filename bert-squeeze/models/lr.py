import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from overrides import overrides

from .base_lt_module import BaseModule


class BowLogisticRegression(BaseModule):
    def __init__(self, vocab_size: int, num_labels: int, training_config: DictConfig, **kwargs):
        super().__init__(training_config, num_labels, **kwargs)
        self.vocab_size = vocab_size

        self._build_model()

    @overrides
    def _build_model(self):
        self.linear = torch.nn.Linear(self.vocab_size, self.num_labels)

    @overrides
    def forward(self, features=None, **kwargs):
        return F.log_softmax(self.linear(features), dim=1)

    def shared_step(self, batch):
        logits = self.forward(**batch)
        loss = self.loss(logits, batch["labels"])
        return loss, logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, logits = self.shared_step(batch)

        self.scorer.add(logits, batch["labels"], loss.detach().cpu())
        if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"loss_{key}"].log(value)

            self.logger.experiment["train/loss"].log(value=logging_loss, step=self.global_step)
            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}


class EmbeddingLogisticRegression(BaseModule):
    def __init__(self, training_config: DictConfig, vocab_size: int, embed_dim: int, num_labels: int,
                 mode: str = "mean", **kwargs):
        super().__init__(training_config, num_labels, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mode = mode
        self._build_model()

    @overrides
    def _build_model(self):
        self.embedding = torch.nn.EmbeddingBag(self.vocab_size, self.embed_dim, mode=self.mode)
        self.pre_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, int(self.embed_dim // 3)),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1)
        )
        self.classifier = torch.nn.Linear(int(self.embed_dim // 3), self.num_labels)

    @overrides
    def forward(self, features=None, offsets=None, **kwargs):
        # for some reason tensors are on CPU, need to move them
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedded = self.embedding(features.to(device), offsets)
        pre_classified = self.pre_classifier(embedded)
        return self.classifier(pre_classified)

    def shared_step(self, batch):
        logits = self.forward(**batch)
        loss = self.loss(logits, batch["labels"])
        return loss, logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, logits = self.shared_step(batch)

        self.scorer.add(logits, batch["labels"], loss.detach().cpu())
        if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"loss_{key}"].log(value)

            self.logger.experiment["train/loss"].log(value=logging_loss, step=self.global_step)
            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}
