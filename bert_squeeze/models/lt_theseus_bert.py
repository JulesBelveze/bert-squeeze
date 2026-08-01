from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoConfig

from bert_squeeze.utils.scorers import Scorer

from ..utils.schedulers.theseus_schedulers import (
    ConstantReplacementScheduler,
    LinearReplacementScheduler,
)
from .base_lt_module import BaseSequenceClassificationTransformerModule
from .custom_transformers import TheseusBertModel


def _should_initialize_successor_layers(
    model: TheseusBertModel, loading_info: object
) -> bool:
    if not isinstance(loading_info, Mapping):
        raise TypeError("Theseus loading information must be a mapping.")

    missing_keys = loading_info.get("missing_keys")
    if not isinstance(missing_keys, list) or not all(
        isinstance(key, str) for key in missing_keys
    ):
        raise TypeError("Theseus loading information must contain missing key names.")

    model_prefix = f"{model.base_model_prefix}."
    normalized_missing_keys = {key.removeprefix(model_prefix) for key in missing_keys}
    expected_successor_keys = {
        f"encoder.successor_layers.{key}"
        for key in model.encoder.successor_layers.state_dict()
    }
    missing_successor_keys = expected_successor_keys & normalized_missing_keys

    if not missing_successor_keys:
        return False
    if missing_successor_keys != expected_successor_keys:
        raise ValueError("Theseus checkpoint has incomplete successor layer weights.")
    return True


class LtTheseusBert(BaseSequenceClassificationTransformerModule):
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
        replacement_scheduler: DictConfig,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        if model is None:
            loaded_model, loading_info = TheseusBertModel.from_pretrained(
                pretrained_model,
                config=AutoConfig.from_pretrained(
                    pretrained_model, num_labels=num_labels
                ),
                output_loading_info=True,
            )
            if not isinstance(loaded_model, TheseusBertModel):
                raise TypeError("Expected a TheseusBertModel checkpoint.")
            model = loaded_model
            if _should_initialize_successor_layers(model, loading_info):
                model.encoder.init_successor_layers()

        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )

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
    def _before_training_step(self) -> None:
        self.replacement_scheduler.step()

    def _build_model(self) -> None:
        self.encoder = self.model

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels),
        )
