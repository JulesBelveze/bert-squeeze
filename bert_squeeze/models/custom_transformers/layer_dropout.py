from typing import Optional

import torch
import torch.nn as nn


class LayerDropoutWrapper(nn.Module):
    """Wraps a transformer layer to enable stochastic layer dropout during training."""

    def __init__(self, layer: nn.Module, dropout_prob: float) -> None:
        super().__init__()
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0 and 1.")
        for name, parameter in layer.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in layer.named_buffers(recurse=False):
            self.register_buffer(
                name,
                buffer,
                persistent=name not in layer._non_persistent_buffers_set,
            )
        for name, module in layer.named_children():
            self.add_module(name, module)
        object.__setattr__(self, "_wrapped_layer", layer)
        self.dropout_prob = dropout_prob

    @property
    def layer(self) -> nn.Module:
        return object.__getattribute__(self, "_wrapped_layer")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, ...]:
        self._sync_wrapped_layer()
        if not self.training or self.dropout_prob == 0.0:
            return self.layer(hidden_states, attention_mask, *args, **kwargs)

        batch_size = hidden_states.shape[0]
        keep_mask = (
            torch.rand(batch_size, device=hidden_states.device) >= self.dropout_prob
        )
        kept_batch_size = int(keep_mask.sum().item())
        if kept_batch_size == batch_size:
            return self.layer(hidden_states, attention_mask, *args, **kwargs)

        if kept_batch_size == 0:
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
        restored_outputs.extend(
            self._restore_batch_value(value, keep_mask, batch_size, kept_batch_size)
            for value in layer_outputs[1:]
        )
        return tuple(restored_outputs)

    def _sync_wrapped_layer(self) -> None:
        self.layer.training = self.training
        for name, parameter in self._parameters.items():
            self.layer._parameters[name] = parameter
        for name, buffer in self._buffers.items():
            self.layer._buffers[name] = buffer

    def _skipped_outputs(
        self,
        hidden_states: torch.Tensor,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[object, ...]:
        hidden_states = self._attach_zero_gradients(hidden_states)
        if not self._output_attentions_enabled(args, kwargs):
            return (hidden_states,)

        attention = getattr(self.layer, "attention", None)
        self_attention = getattr(attention, "self", None)
        num_heads = getattr(self_attention, "num_attention_heads", None)
        if not isinstance(num_heads, int):
            raise AttributeError("Wrapped layer does not expose its attention heads.")
        sequence_length = hidden_states.shape[1]
        attentions = hidden_states.new_zeros(
            hidden_states.shape[0], num_heads, sequence_length, sequence_length
        )
        return hidden_states, attentions

    def _attach_zero_gradients(self, hidden_states: torch.Tensor) -> torch.Tensor:
        zero_dependency = hidden_states.new_zeros(())
        for parameter in self.parameters():
            if parameter.numel() > 0:
                zero_dependency = zero_dependency + parameter.reshape(-1)[0] * 0.0
        return hidden_states + zero_dependency

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
