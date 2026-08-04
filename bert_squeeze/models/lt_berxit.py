from __future__ import annotations

import logging
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
from .custom_transformers.berxit import BerxitModel


class LtBerxit(BaseSequenceClassificationTransformerModule):
    """Fine-tune BERxiT models for sequence classification."""

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
            model = BerxitModel.from_pretrained(
                pretrained_model,
                config=AutoConfig.from_pretrained(
                    pretrained_model, num_labels=num_labels
                ),
            )

        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )
        self.train_stage = getattr(training_config, "train_stage", "backbone")
        self.switch_step: Optional[int] = getattr(training_config, "switch_step", None)
        self.train_highway = training_config.train_highway
        self.train_gates = self.train_stage == "gates" or getattr(
            training_config, "train_gates", False
        )
        self._build_model()
        if self.train_stage == "gates":
            self._freeze_backbone_for_gates()
        self._has_switched_stage = False

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Sequence[RampOutput],
        int,
        Optional[Tuple[torch.Tensor, ...]],
    ]:
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
            gates_logits = outputs.gates_logits
        else:
            ramps_exits = outputs.ramps_exits
            exit_layer = outputs.exit_layer
            logits = outputs.logits
            gates_logits = None

        return logits, ramps_exits, exit_layer, gates_logits

    @overrides
    def _classification_output(
        self, batch: Dict[str, torch.Tensor]
    ) -> SequenceClassificationOutput:
        logits, ramps_exits, exit_layer, gates_logits = self.forward(
            **self._model_inputs(batch)
        )
        return SequenceClassificationOutput(
            logits=logits,
            ramps_exits=ramps_exits,
            gates_logits=gates_logits,
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
            train_ramps=self._train_all_exits(),
            ramps_exits=output.ramps_exits,
            train_gates=self.train_gates,
            gates_logits=output.gates_logits,
        )

    @overrides
    def _before_training_step(self) -> None:
        self._maybe_switch_stage()

    @overrides
    def _before_prediction_step(self) -> None:
        self.bert.set_inference_mode(inference=True)

    def _train_all_exits(self) -> bool:
        return self.train_highway and self.global_step % 2 == 1

    def _freeze_backbone_for_gates(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = "gates" in name

    def _maybe_switch_stage(self) -> None:
        if (
            self.switch_step is None
            or self.train_stage == "gates"
            or self._has_switched_stage
        ):
            return

        if self.global_step >= self.switch_step:
            logging.info(
                "Switching LtBerxit training stage from 'backbone' to 'gates' at "
                f"global_step={self.global_step}"
            )
            self.train_stage = "gates"
            self.train_gates = True
            self._freeze_backbone_for_gates()
            if self.trainer is not None:
                self.trainer.strategy.setup_optimizers(self.trainer)
            self._has_switched_stage = True

    @overrides
    def _get_optimizer_parameters(self) -> List[OptimizerParameterGroup]:
        if getattr(self, "train_stage", "backbone") == "gates":
            named_parameters = (
                (name, parameter)
                for name, parameter in self.named_parameters()
                if "gates" in name and parameter.requires_grad
            )
            return build_optimizer_parameter_groups(
                named_parameters,
                discriminative_learning=False,
                learning_rates=self.config.learning_rates,
                layer_lr_decay=self.config.get("layer_lr_decay", 1.0),
                weight_decay=self.config.weight_decay,
            )

        discriminative_learning = self.config.discriminative_learning
        named_parameters = (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
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
        train_gates: bool = False,
        gates_logits: Optional[Tuple[torch.Tensor, ...]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if ramps_exits is None:
            if logits is None:
                raise ValueError("BERxiT training requires classifier outputs.")
            return CrossEntropyLoss()(
                logits.view(-1, self.model_config.num_labels), labels.view(-1)
            )

        exit_indices = (
            tuple(range(len(ramps_exits))) if train_ramps else (len(ramps_exits) - 1,)
        )
        loss_fct = CrossEntropyLoss()
        classification_losses = [
            loss_fct(
                ramps_exits[index].logits.view(-1, self.model_config.num_labels),
                labels.view(-1),
            )
            for index in exit_indices
        ]
        loss = torch.stack(classification_losses).sum()

        if not train_gates:
            return loss
        if gates_logits is None or len(gates_logits) != len(ramps_exits):
            raise ValueError("Gate training requires one gate output per ramp.")
        return loss + self._gate_loss(labels, ramps_exits, gates_logits, exit_indices)

    @staticmethod
    def _gate_loss(
        labels: torch.Tensor,
        ramps_exits: Sequence[RampOutput],
        gates_logits: Sequence[torch.Tensor],
        exit_indices: Sequence[int],
    ) -> torch.Tensor:
        gate_losses = []
        for index in exit_indices:
            prediction = ramps_exits[index].logits.argmax(dim=-1)
            target = (prediction == labels).float()
            certainty = torch.sigmoid(gates_logits[index]).squeeze(-1)
            gate_losses.append(torch.nn.functional.mse_loss(certainty, target))
        return torch.stack(gate_losses).sum()

    def _build_model(self) -> None:
        self.bert = self.model
        self.num_layers = len(self.bert.encoder.layer)
        self.bert.encoder.set_early_exit_entropy(self.config.early_exit_entropy)
        if hasattr(self.config, "gate_thresholds"):
            self.bert.set_exit_gate_thresholds(self.config.gate_thresholds)
        self.bert.init_highway_pooler()
