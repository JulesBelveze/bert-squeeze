from typing import MutableMapping, Optional

import torch
import torch.nn as nn


class LayerDropoutWrapper(nn.Module):
    """Wraps a transformer layer to enable stochastic layer dropout during training."""

    def __init__(self, layer: nn.Module, dropout_prob: float) -> None:
        super().__init__()
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0 and 1.")
        self.layer = layer
        self.dropout_prob = dropout_prob

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, ...]:
        if not self.training or self.dropout_prob == 0.0:
            return self.layer(hidden_states, attention_mask, *args, **kwargs)

        batch_size = hidden_states.shape[0]
        keep_mask = (
            torch.rand(batch_size, device=hidden_states.device) >= self.dropout_prob
        )
        if keep_mask.all():
            return self.layer(hidden_states, attention_mask, *args, **kwargs)

        if not keep_mask.any():
            return self._skipped_outputs(hidden_states, args, kwargs)

        kept_attention_mask = self._slice_batch_value(
            attention_mask, keep_mask, batch_size
        )
        kept_args = tuple(
            self._slice_batch_value(value, keep_mask, batch_size) for value in args
        )
        kept_kwargs = {
            key: self._slice_batch_value(value, keep_mask, batch_size)
            for key, value in kwargs.items()
        }
        layer_outputs = self.layer(
            hidden_states[keep_mask], kept_attention_mask, *kept_args, **kept_kwargs
        )
        if not isinstance(layer_outputs, tuple):
            raise TypeError("LayerSkip transformer layers must return a tuple.")

        restored_hidden_states = hidden_states.clone()
        restored_hidden_states[keep_mask] = layer_outputs[0]
        restored_outputs: list[object] = [restored_hidden_states]
        kept_batch_size = layer_outputs[0].shape[0]
        restored_outputs.extend(
            self._restore_batch_value(value, keep_mask, batch_size, kept_batch_size)
            for value in layer_outputs[1:]
        )
        return tuple(restored_outputs)

    def state_dict(
        self,
        *args: object,
        destination: Optional[MutableMapping[str, torch.Tensor]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> MutableMapping[str, torch.Tensor]:
        return self.layer.state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def _load_from_state_dict(
        self,
        state_dict: MutableMapping[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        wrapped_prefix = f"{prefix}layer."
        for layer_key in self.layer.state_dict():
            public_key = f"{prefix}{layer_key}"
            wrapped_key = f"{wrapped_prefix}{layer_key}"
            if public_key in state_dict and wrapped_key not in state_dict:
                state_dict[wrapped_key] = state_dict.pop(public_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _skipped_outputs(
        self,
        hidden_states: torch.Tensor,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[object, ...]:
        if not self._output_attentions_enabled(args, kwargs):
            return (hidden_states,)

        num_heads = self.layer.attention.self.num_attention_heads
        sequence_length = hidden_states.shape[1]
        attentions = hidden_states.new_zeros(
            hidden_states.shape[0], num_heads, sequence_length, sequence_length
        )
        return hidden_states, attentions

    @staticmethod
    def _output_attentions_enabled(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> bool:
        configured_value = kwargs.get("output_attentions")
        if isinstance(configured_value, bool):
            return configured_value
        return len(args) >= 5 and args[4] is True

    @staticmethod
    def _slice_batch_value(
        value: object, keep_mask: torch.Tensor, batch_size: int
    ) -> object:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] == batch_size:
                return value[keep_mask]
        if isinstance(value, tuple):
            return tuple(
                LayerDropoutWrapper._slice_batch_value(item, keep_mask, batch_size)
                for item in value
            )
        return value

    @staticmethod
    def _restore_batch_value(
        value: object,
        keep_mask: torch.Tensor,
        batch_size: int,
        kept_batch_size: int,
    ) -> object:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] == kept_batch_size:
                restored = value.new_zeros((batch_size, *value.shape[1:]))
                restored[keep_mask] = value
                return restored
        if isinstance(value, tuple):
            return tuple(
                LayerDropoutWrapper._restore_batch_value(
                    item, keep_mask, batch_size, kept_batch_size
                )
                for item in value
            )
        return value
