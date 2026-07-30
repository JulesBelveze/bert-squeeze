from __future__ import annotations

from typing import Tuple, Union

import torch
from overrides import overrides

from .base_lt_module import BaseSequenceClassificationTransformerModule


class LtCustomDistilBert(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune a DistilBERT based model on a sequence classification task.

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

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
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
            Logits, optionally paired with per-layer attention tensors.
        """
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("DistilBERT sequence classifiers must return tensor logits.")

        if output_attentions:
            attentions = getattr(outputs, "attentions", None)
            if not isinstance(attentions, tuple):
                raise TypeError(
                    "DistilBERT did not return attentions when they were requested."
                )
            return logits, attentions
        return logits

    def _classification_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("DistilBERT classification must return tensor logits.")
        return outputs

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        logits = self._classification_logits(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(labels=batch["labels"], logits=logits)

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
        logits = self._classification_logits(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(labels=batch["labels"], logits=logits)

        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        logits = self._classification_logits(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(labels=batch["labels"], logits=logits)

        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss
