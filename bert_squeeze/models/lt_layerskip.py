from typing import Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from overrides import overrides

from bert_squeeze.utils.scorers import BaseSequenceClassificationScorer

from .base_lt_module import BaseSequenceClassificationTransformerModule
from .custom_transformers.layer_dropout import LayerDropoutWrapper


class LtLayerSkip(BaseSequenceClassificationTransformerModule):
    """Fine-tunes BERT classifiers and exits after ``exit_layer`` encoder blocks."""

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
        scorer: Optional[BaseSequenceClassificationScorer] = None,
    ):
        if not 0.0 <= p_max <= 1.0:
            raise ValueError("p_max must be between 0 and 1.")
        if e_scale < 0.0:
            raise ValueError("e_scale must be non-negative.")
        if dropout_schedule not in {"exponential", "linear", "uniform"}:
            raise ValueError(
                "dropout_schedule must be one of: exponential, linear, uniform."
            )

        if scorer is None:
            super().__init__(
                training_config=training_config,
                pretrained_model=pretrained_model,
                num_labels=num_labels,
                model=model,
            )
        else:
            super().__init__(
                training_config=training_config,
                pretrained_model=pretrained_model,
                num_labels=num_labels,
                model=model,
                scorer=scorer,
            )
        if self.model_config.model_type != "bert":
            raise ValueError("LtLayerSkip currently supports BERT models only.")
        self.p_max = p_max
        self.e_scale = e_scale
        self.dropout_schedule = dropout_schedule
        self.inference_mode = inference_mode

        layers = self._get_transformer_layers()
        self.num_layers = len(layers)
        self.exit_layer = (
            exit_layer if exit_layer is not None else max(1, self.num_layers // 2)
        )
        if not 1 <= self.exit_layer <= self.num_layers:
            raise ValueError(
                "exit_layer must be in [1, num_layers], got "
                f"{self.exit_layer} for num_layers={self.num_layers}."
            )

        self._wrap_layers_with_dropout()
        self.register_buffer("curriculum_mask", torch.ones(self.num_layers))
        self.register_buffer("loss_scales", self._compute_loss_scales())

    @overrides
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.inference_mode and not self.training:
            hidden_states = self._forward_early_exit(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        else:
            hidden_states = self._forward_all_layers(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )[-1]
        return self._get_layer_logits(hidden_states)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        inputs = self._build_inputs(batch)
        outputs = self._forward_all_layers(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        loss = self._compute_training_loss(outputs, batch["labels"])

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

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return self._eval_step(batch, self.valid_scorer, self.validation_step_outputs)

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return self._eval_step(batch, self.test_scorer, self.test_step_outputs)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        inputs = self._build_inputs(batch)
        logits = self.forward(**inputs)
        return torch.softmax(logits, dim=-1)

    def _compute_training_loss(
        self, outputs: tuple[torch.Tensor, ...], labels: torch.Tensor
    ) -> torch.Tensor:
        weights = self.loss_scales * self.curriculum_mask
        enabled_layers = torch.nonzero(weights > 0, as_tuple=False).flatten()
        if enabled_layers.numel() == 0:
            enabled_layers = torch.tensor(
                [self.num_layers - 1], device=weights.device, dtype=torch.long
            )
            weights = weights.clone()
            weights[-1] = 1.0

        base_loss = super().loss
        layer_losses = torch.stack(
            [
                base_loss(
                    labels=labels,
                    logits=self._get_layer_logits(outputs[layer_idx]),
                )
                for layer_idx in enabled_layers.tolist()
            ]
        )
        enabled_weights = weights[enabled_layers]
        return (layer_losses * enabled_weights).sum() / enabled_weights.sum()

    def _forward_all_layers(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
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
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            **forward_kwargs,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("output_hidden_states must be enabled for LayerSkip.")
        return tuple(hidden_states[1:])

    def _eval_step(
        self,
        batch: dict[str, torch.Tensor],
        scorer: BaseSequenceClassificationScorer,
        output_store: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        inputs = self._build_inputs(batch)
        logits = self.forward(**inputs)
        labels = batch["labels"]
        loss = super().loss(labels=labels, logits=logits)
        scorer.add(logits.cpu(), labels.cpu(), loss.cpu())
        output_store.append(
            {"loss": loss, "logits": logits.cpu(), "labels": labels.cpu()}
        )
        return loss

    @staticmethod
    def _build_inputs(
        batch: dict[str, torch.Tensor]
    ) -> dict[str, Optional[torch.Tensor]]:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch.get("token_type_ids"),
        }

    def _forward_early_exit(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids or inputs_embeds are required for early exit.")

        base_model = self._get_base_model()
        batch_size, sequence_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            buffered_token_type_ids = getattr(
                base_model.embeddings, "token_type_ids", None
            )
            if buffered_token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            else:
                token_type_ids = buffered_token_type_ids[:, :sequence_length].expand(
                    batch_size, sequence_length
                )

        hidden_states = base_model.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        extended_attention_mask = base_model.get_extended_attention_mask(
            attention_mask, input_shape
        )
        prepared_head_mask = base_model.get_head_mask(head_mask, self.num_layers)

        layers = self._get_transformer_layers()
        for layer_idx in range(self.exit_layer):
            layer_outputs = layers[layer_idx](
                hidden_states,
                extended_attention_mask,
                prepared_head_mask[layer_idx],
            )
            hidden_states = layer_outputs[0]
        return hidden_states

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
            layers[idx] = LayerDropoutWrapper(layer, prob)

    def _compute_dropout_schedule(self) -> list[float]:
        if self.num_layers <= 1:
            return [0.0] * self.num_layers

        if self.dropout_schedule == "exponential":
            return [
                self.p_max * (2 ** (layer_idx / (self.num_layers - 1)) - 1.0)
                for layer_idx in range(self.num_layers)
            ]
        if self.dropout_schedule == "linear":
            return [
                self.p_max * layer_idx / (self.num_layers - 1)
                for layer_idx in range(self.num_layers)
            ]
        if self.dropout_schedule == "uniform":
            return [self.p_max] * self.num_layers

        raise RuntimeError("Unsupported dropout schedule.")

    def _compute_loss_scales(self) -> torch.Tensor:
        if self.num_layers <= 1:
            return torch.ones(self.num_layers)

        early_scales = [
            self.e_scale * (layer_idx * (layer_idx + 1) / 2)
            for layer_idx in range(self.num_layers - 1)
        ]
        previous_scale_sum = (self.num_layers - 2) * (self.num_layers - 1) / 2
        final_scale = self.num_layers - 1 + self.e_scale * previous_scale_sum
        scales = torch.tensor([*early_scales, final_scale], dtype=torch.float)
        return scales / scales.sum()
