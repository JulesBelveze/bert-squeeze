import torch
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoConfig

from ..utils.schedulers.theseus_schedulers import (
    ConstantReplacementScheduler,
    LinearReplacementScheduler,
)
from .base_lt_module import BaseTransformerModule
from .custom_transformers import TheseusBertModel


class LtTheseusBert(BaseTransformerModule):
    """
    Lightning module to fine-tune a TheseusBert based model on a sequence classification
    task (see `models.custom_transformers.theseus_bert.py`) for detailed explanation.

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels
        pretrained_model (str):
            name of the pretrained Transformer model to use
        replacement_scheduler (DictConfig):
            configuration for the replacement scheduler
    """

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        replacement_scheduler: DictConfig,
        **kwargs,
    ):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)

        self._build_model()
        scheduler = {
            "linear": LinearReplacementScheduler,
            "constant": ConstantReplacementScheduler,
        }[replacement_scheduler.type]

        self.replacement_scheduler = scheduler(
            self.encoder.encoder,
            **{k: v for k, v in replacement_scheduler.items() if k != "type"},
        )

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor):
                sentence or sentences represented as tokens
            attention_mask (torch.Tensor):
                tells the model which tokens in the input_ids are words and which are padding.
                               1 indicates a token and 0 indicates padding.
            token_type_ids (torch.Tensor):
                used when there are two sentences that need to be part of the input. It indicate which
                               tokens are part of sentence1 and which are part of sentence2.
            position_ids (torch.Tensor):
                indices of positions of each input sequence tokens in the position embeddings. Selected
                             in the range ``[0, config.max_position_embeddings - 1]
            head_mask (torch.Tensor):
                mask to nullify selected heads of the self-attention modules
        Returns:
            torch.Tensor: predicted logits
        """
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        self.replacement_scheduler.step()

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"])

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss.detach().cpu())
        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {
                key: torch.stack(val).mean() for key, val in self.scorer.losses.items()
            }
            self.log_dict({f"train/loss_{key}": val for key, val in logging_loss.items()})
            self.log("train/acc", self.scorer.acc)
            self.scorer.reset()

        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> None:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"])
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> None:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"])
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )

    @overrides
    def _build_model(self):
        """"""
        encoder = TheseusBertModel(AutoConfig.from_pretrained(self.pretrained_model))
        encoder.from_pretrained(self.pretrained_model)
        encoder.encoder.init_successor_layers()
        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels),
        )
