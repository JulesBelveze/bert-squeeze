from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class LayerDropoutWrapper(nn.Module):
    """Wraps a transformer layer to enable stochastic layer dropout during training."""

    def __init__(self, layer: nn.Module, dropout_prob: float, layer_idx: int) -> None:
        super().__init__()
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: object,
        **kwargs: object,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if not self.training or self.dropout_prob == 0.0:
            return self.layer(hidden_states, attention_mask, *args, **kwargs)

        if torch.rand(1, device=hidden_states.device).item() > self.dropout_prob:
            return self.layer(hidden_states, attention_mask, *args, **kwargs)

        output = self.layer(hidden_states, attention_mask, *args, **kwargs)
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
