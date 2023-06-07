from typing import Tuple, Union

import torch
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoModel

from .base_lt_module import BaseTransformerModule


class LtCustomDistilBert(BaseTransformerModule):
    """
    Lightning module to fine-tune a DistilBERT based model on a sequence classification task.

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels
        pretrained_model (str):
            name of the pretrained Transformer model to use
    """

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        **kwargs,
    ):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)
        self._build_model()

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input_ids (torch.Tensor):
                sentence or sentences represented as tokens
            attention_mask (torch.Tensor):
                tells the model which tokens in the input_ids are words and which are padding.
                1 indicates a token and 0 indicates padding.
            output_attentions (bool):
                whether to output attention scores.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: logits obtained from model pass
                along with the attention scores if `output_attentions=True`.
        """
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_state = outputs[0]
        logits = self.classifier(hidden_state)

        if output_attentions:
            return logits, outputs.attentions
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        logits = self.forward(**inputs)
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

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        logits = self.forward(**inputs)
        loss = self.loss(logits, batch["labels"])

        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        logits = self.forward(**inputs)
        loss = self.loss(logits, batch["labels"])

        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def _build_model(self):
        """"""
        self.encoder = AutoModel.from_pretrained(self.pretrained_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.seq_classif_dropout),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels),
        )
