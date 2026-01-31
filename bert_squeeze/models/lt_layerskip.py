import logging
import math
from typing import List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from overrides import overrides

from bert_squeeze.utils.scorers import Scorer

from .base_lt_module import BaseSequenceClassificationTransformerModule
from .custom_transformers.layer_dropout import LayerDropoutWrapper


class LtLayerSkip(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune a LayerSkip-style model on sequence classification
    tasks with layer dropout and early exit supervision.
    """

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        p_max: float = 0.1,
        e_scale: float = 0.2,
        exit_layer: Optional[int] = None,
        dropout_schedule: str = "exponential",
        inference_mode: bool = False,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs: object,
    ):
        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )
        self.p_max = p_max
        self.e_scale = e_scale
        self.dropout_schedule = dropout_schedule
        self.inference_mode = inference_mode

        layers = self._get_transformer_layers()
        self.num_layers = len(layers)
        self.exit_layer = exit_layer if exit_layer is not None else self.num_layers // 2
        if not 0 <= self.exit_layer < self.num_layers:
            raise ValueError(
                "exit_layer must be in [0, num_layers - 1], got "
                f"{self.exit_layer} for num_layers={self.num_layers}."
            )

        self._wrap_layers_with_dropout()
        self.register_buffer("curriculum_mask", torch.ones(self.num_layers))
        self.register_buffer("loss_scales", self._compute_loss_scales())

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        if self.training or not self.inference_mode:
            return self._forward_all_layers(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                **kwargs,
            )
        return self._forward_early_exit(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            **kwargs,
        )

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch.get("token_type_ids"),
        }
        outputs = self.forward(**inputs)
        loss = self.loss(outputs=outputs, labels=batch["labels"])

        logits = self._get_layer_logits(outputs[-1])
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
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch.get("token_type_ids"),
        }

        if self.inference_mode:
            hidden_states = self._forward_early_exit(**inputs)
        else:
            hidden_states = self._forward_all_layers(**inputs)[-1]

        logits = self._get_layer_logits(hidden_states)
        loss = super().loss(labels=batch["labels"], logits=logits)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch.get("token_type_ids"),
        }

        if self.inference_mode:
            hidden_states = self._forward_early_exit(**inputs)
        else:
            hidden_states = self._forward_all_layers(**inputs)[-1]

        logits = self._get_layer_logits(hidden_states)
        loss = super().loss(labels=batch["labels"], logits=logits)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def predict_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch.get("token_type_ids"),
        }
        if self.inference_mode:
            hidden_states = self._forward_early_exit(**inputs)
        else:
            hidden_states = self._forward_all_layers(**inputs)[-1]

        logits = self._get_layer_logits(hidden_states)
        return torch.softmax(logits, dim=-1)

    def loss(
        self, outputs: Tuple[torch.Tensor, ...], labels: torch.Tensor
    ) -> torch.Tensor:
        layer_losses = []
        for hidden_states in outputs:
            logits = self._get_layer_logits(hidden_states)
            layer_losses.append(super().loss(labels=labels, logits=logits))

        losses = torch.stack(layer_losses)
        weights = self.loss_scales * self.curriculum_mask
        weight_sum = weights.sum()
        if weight_sum.item() <= 0:
            return losses[-1]
        return (losses * weights).sum() / weight_sum

    def _forward_all_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, ...]:
        base_model = self._get_base_model()
        forward_kwargs = dict(kwargs)
        forward_kwargs.pop("output_hidden_states", None)
        forward_kwargs.pop("return_dict", None)
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_hidden_states=True,
            return_dict=True,
            **forward_kwargs,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("output_hidden_states must be enabled for LayerSkip.")
        return tuple(hidden_states[1:])

    def _forward_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = self._forward_all_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            **kwargs,
        )
        return hidden_states[self.exit_layer]

    def _get_layer_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = self._pool_hidden_states(hidden_states)
        dropout = getattr(self.model, "dropout", None)
        if isinstance(dropout, nn.Module):
            pooled_output = dropout(pooled_output)

        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, nn.Module):
            raise AttributeError("Model classifier head not found for LayerSkip.")
        return classifier(pooled_output)

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_model = self._get_base_model()
        pooler = getattr(base_model, "pooler", None)
        if isinstance(pooler, nn.Module):
            return pooler(hidden_states)
        return hidden_states[:, 0, :]

    def _get_base_model(self) -> nn.Module:
        base_prefix = getattr(self.model, "base_model_prefix", None)
        if base_prefix is None:
            raise ValueError("Model base prefix not found; expected a HF model.")
        base_model = getattr(self.model, base_prefix, None)
        if base_model is None:
            raise ValueError(f"Model has no attribute '{base_prefix}'.")
        return base_model

    def _get_transformer_layers(self) -> nn.ModuleList:
        base_model = self._get_base_model()
        encoder = getattr(base_model, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            raise ValueError("Unable to locate transformer layers for LayerSkip.")
        layers = encoder.layer
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("Transformer layers must be a ModuleList.")
        return layers

    def _wrap_layers_with_dropout(self) -> None:
        layers = self._get_transformer_layers()
        dropout_probs = self._compute_dropout_schedule()
        if len(dropout_probs) != len(layers):
            raise ValueError("Dropout schedule does not match number of layers.")

        for idx, (layer, prob) in enumerate(zip(layers, dropout_probs)):
            layers[idx] = LayerDropoutWrapper(layer, prob, idx)

    def _compute_dropout_schedule(self) -> List[float]:
        if self.num_layers <= 1:
            return [0.0] * self.num_layers

        if self.dropout_schedule == "exponential":
            scale = math.log(2.0) / (self.num_layers - 1)
            return [
                min(self.p_max, self.p_max * (math.exp(layer_idx * scale) - 1.0))
                for layer_idx in range(self.num_layers)
            ]
        if self.dropout_schedule == "linear":
            return [
                self.p_max * layer_idx / (self.num_layers - 1)
                for layer_idx in range(self.num_layers)
            ]
        if self.dropout_schedule == "uniform":
            return [self.p_max] * self.num_layers

        raise ValueError(
            "dropout_schedule must be one of: exponential, linear, uniform. Got "
            f"{self.dropout_schedule}."
        )

    def _compute_loss_scales(self) -> torch.Tensor:
        if self.num_layers <= 1:
            return torch.ones(self.num_layers)

        scales: List[float] = []
        for layer_idx in range(self.num_layers):
            if layer_idx < self.num_layers - 1:
                scale = self.e_scale * (layer_idx * (layer_idx + 1) / 2)
            else:
                prev_sum = (self.num_layers - 2) * (self.num_layers - 1) / 2
                scale = (self.num_layers - 1) + self.e_scale * prev_sum
            scales.append(scale)

        total = sum(scales)
        if total <= 0:
            logging.warning("Loss scales sum to 0; falling back to uniform weights.")
            return torch.ones(self.num_layers) / self.num_layers
        return torch.tensor(scales, dtype=torch.float) / total
