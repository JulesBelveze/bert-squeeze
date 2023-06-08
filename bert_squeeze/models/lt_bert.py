from typing import Tuple, Union

import torch
from omegaconf import DictConfig
from overrides import overrides

from .base_lt_module import BaseTransformerModule
from .custom_transformers import CustomBertModel


class LtCustomBert(BaseTransformerModule):
    """
    Lightning module to fine-tune a BERT based model on a sequence classification task.

    Note: It uses a custom BERT model (see `models.custom_transformers.bert.py`) for
    explanation.

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels for the classification tasks
        pretrained_model (str):
            name of the pretrained Transformer model to use
    """

    def __init__(
        self,
        training_config: DictConfig,
        num_labels: int,
        pretrained_model: str,
        **kwargs,
    ):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)
        self._build_model()

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
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
            token_type_ids (torch.Tensor):
                used when there are two sentences that need to be part of the input. It indicates which
                tokens are part of sentence1 and which are part of sentence2.
            output_attentions (bool):
                whether to output attention scores.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: logits obtained from model pass
                along with the attention scores if `output_attentions=True`.

        For specifications about model output, please refer to:
        https://github.com/huggingface/transformers/blob/b01f451ca38695c60175b34d245997ef4d18231d/src/transformers/modeling_outputs.py#L153
        """
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        if output_attentions:
            return logits, outputs.attentions
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
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
            self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})
            self.log("train/acc", self.scorer.acc)
            self.scorer.reset()
        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"].float())

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
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"])
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def _build_model(self):
        """"""
        self.encoder = CustomBertModel.from_pretrained(self.pretrained_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels),
        )
