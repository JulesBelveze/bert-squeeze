from __future__ import annotations

from typing import List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from adapters import AutoAdapterModel, ModelWithFlexibleHeadsAdaptersMixin, Stack
from omegaconf import DictConfig
from overrides import overrides

from bert_squeeze.utils.scorers import Scorer

from .base_lt_module import BaseSequenceClassificationTransformerModule


class LtAdapter(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune adapters for Transformer-based language models on sequence classification task.

    Note: It uses the adapters library under the hood, check it out to better understand how to choose parameters
    https://github.com/adapter-hub/adapters

    Args:
        training_config (DictConfig):
            training configuration
        pretrained_model (str):
            name of the pretrained Transformer model to use
        task_name (str):
            name for the adapter configuration
        adapter_config_name (str):
            nam of the adapter config to use
        labels (Union[List[str], List[int]]):
            list of labels used for the classification head
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        scorer (Scorer):
            helper object to compute performance metrics during training
    """

    BASE_CLASS_MODEL = AutoAdapterModel

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        task_name: str,
        adapter_config_name: str,
        labels: Union[List[str], List[int]],
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        super().__init__(
            training_config, pretrained_model, len(labels), model, scorer, **kwargs
        )

        assert len(labels) == self.model_config.num_labels

        self.task_name = task_name
        self.adapter_config_name = adapter_config_name
        self.labels = labels

        self._configure_adapter()

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
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
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: logits obtained from model pass

        For specifications about model output, please refer to:
        https://github.com/huggingface/transformers/blob/b01f451ca38695c60175b34d245997ef4d18231d/src/transformers/modeling_outputs.py#L153
        """
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return outputs.logits

    def _configure_adapter(self) -> None:
        model = self.model
        if not isinstance(model, ModelWithFlexibleHeadsAdaptersMixin):
            raise TypeError("LtAdapter requires a model with flexible adapter heads.")

        if self.task_name not in model.adapters_config:
            model.add_adapter(self.task_name, config=self.adapter_config_name)
        if self.task_name not in model.heads:
            model.add_classification_head(
                head_name=self.task_name,
                num_labels=self.model_config.num_labels,
                id2label={i: label for i, label in enumerate(self.labels)},
            )
        model.train_adapter(Stack(self.task_name))
