from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from overrides import overrides
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig

from bert_squeeze.utils.optimizers import (
    OptimizerParameterGroup,
    build_optimizer_parameter_groups,
)
from bert_squeeze.utils.scorers import Scorer
from bert_squeeze.utils.types import RampOutput, SequenceClassificationOutput

from .base_lt_module import BaseSequenceClassificationTransformerModule
from .custom_transformers.deebert import DeeBertModel


class LtDeeBert(BaseSequenceClassificationTransformerModule):
    """Fine-tune DeeBERT models for sequence classification."""

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        if model is None:
            model = DeeBertModel.from_pretrained(
                pretrained_model,
                config=AutoConfig.from_pretrained(
                    pretrained_model, num_labels=num_labels
                ),
            )

        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )
        self.train_highway = training_config.train_highway
        self._build_model()

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Sequence[RampOutput], int]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        if not self.bert.encoder.inference:
            exit_layer = self.num_layers
            ramps_exits = outputs.ramps_exits
            logits = ramps_exits[-1].logits
        else:
            ramps_exits = outputs.ramps_exits
            exit_layer = outputs.exit_layer
            logits = outputs.logits

        return logits, ramps_exits, exit_layer

    @overrides
    def _classification_output(
        self, batch: Dict[str, torch.Tensor]
    ) -> SequenceClassificationOutput:
        logits, ramps_exits, exit_layer = self.forward(**self._model_inputs(batch))
        return SequenceClassificationOutput(
            logits=logits,
            ramps_exits=ramps_exits,
            exit_layer=exit_layer,
        )

    @overrides
    def _classification_loss(
        self,
        output: SequenceClassificationOutput,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(
            logits=output.logits,
            labels=labels,
            train_ramps=self.train_highway,
            ramps_exits=output.ramps_exits,
        )

    @overrides
    def _before_prediction_step(self) -> None:
        self.bert.set_inference_mode(inference=True)

    @overrides
    def _get_optimizer_parameters(self) -> List[OptimizerParameterGroup]:
        named_parameters = (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
        discriminative_learning = (
            self.config.discriminative_learning and not self.train_highway
        )
        return build_optimizer_parameter_groups(
            named_parameters,
            discriminative_learning=discriminative_learning,
            learning_rates=self.config.learning_rates,
            layer_lr_decay=self.config.get("layer_lr_decay", 1.0),
            weight_decay=self.config.weight_decay,
        )

    @overrides
    def loss(
        self,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        ramps_exits: Optional[Sequence[RampOutput]] = None,
        train_ramps: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        loss_fct = CrossEntropyLoss()
        if train_ramps:
            if ramps_exits is None or len(ramps_exits) < 2:
                raise ValueError("Ramp training requires at least two ramp outputs.")
            return torch.stack(
                [
                    loss_fct(
                        ramp.logits.view(-1, self.model_config.num_labels),
                        labels.view(-1),
                    )
                    for ramp in ramps_exits[:-1]
                ]
            ).sum()

        if logits is None:
            raise ValueError("Classifier logits are required when ramps are disabled.")
        return loss_fct(logits.view(-1, self.model_config.num_labels), labels.view(-1))

    def _build_model(self) -> None:
        self.bert = self.model
        self.num_layers = len(self.bert.encoder.layer)
        self.bert.encoder.set_early_exit_entropy(self.config.early_exit_entropy)
        self.bert.init_highway_pooler()
        self._set_trainable_parameters()

    def _set_trainable_parameters(self) -> None:
        final_ramp = f".ramp.{self.num_layers - 1}."
        for name, parameter in self.named_parameters():
            is_ramp = ".ramp." in name
            is_final_ramp = final_ramp in name
            if self.train_highway:
                trainable = is_ramp and not is_final_ramp
            else:
                trainable = not is_ramp or is_final_ramp

            if ".pooler." in name and not is_ramp:
                trainable = False
            parameter.requires_grad = trainable
