from __future__ import annotations

from typing import Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from overrides import overrides

from bert_squeeze.utils.scorers import Scorer

from .base_lt_module import BaseSequenceClassificationTransformerModule


class LtSequenceClassificationAutoModel(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune any Hugging Face encoder on a sequence classification
    task through ``AutoModelForSequenceClassification``.

    Unlike `LtSequenceClassificationCustomBert`, which is tied to the BERT architecture,
    this module relies on the architecture-native classification head, so it supports any
    encoder exposed by the ``transformers`` auto classes (ModernBERT, RoBERTa, DeBERTa,
    DistilBERT, ...). It therefore works both as a standalone classifier and as a teacher
    or student in `DistilAssistant`.

    Args:
        training_config (DictConfig):
            training configuration
        pretrained_model (str):
            name of the pretrained Transformer model to use
        num_labels (int):
            number of labels for the classification task
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        scorer (Scorer):
            helper object to compute performance metrics during training
    """

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )
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
                tells the model which tokens are words (1) and which are padding (0)
            token_type_ids (torch.Tensor):
                segment ids; only forwarded when provided as some encoders (e.g.
                ModernBERT, RoBERTa) do not accept them
            output_attentions (bool):
                whether to output attention scores.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: logits, along with the
                attention scores when `output_attentions=True`.
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(**model_inputs)
        if output_attentions:
            return outputs.logits, outputs.attentions
        return outputs.logits

    def _build_model(self):
        """"""
        # `base_model` is the backbone without the classification head, so freezing it
        # keeps the head trainable (see `freeze_encoder`).
        self.encoder = self.model.base_model
