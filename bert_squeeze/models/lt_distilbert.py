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

    @overrides
    def _model_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_names = ("input_ids", "attention_mask")
        return {name: batch[name] for name in input_names if name in batch}
