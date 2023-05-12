import logging
import random
from abc import ABC
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert import BertLayer, BertModel


class BertCustomEncoder(nn.Module):
    """
    Custom BERT encoder to enable layer dropping according to various techniques.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture.
    """

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layerdrop = kwargs.get("layerdrop", 0.0)

        if kwargs.get("poor_man_technique", None) is not None:
            self.layer_to_drop = self.get_layer_to_prune(
                kwargs["poor_man_technique"], kwargs["K"]
            )
        else:
            self.layer_to_drop = None

        self.gradient_checkpointing = False

    def get_layer_to_prune(self, dropping_technique: str, K: int) -> List[bool]:
        """
        Implementation of various layer dropping strategies as presented in
        https://arxiv.org/abs/2004.03844.

        Args:
            dropping_technique (str):
                dropping strategy to use to prune layers. The following strategies
                are available:
                - top: dropping the top K layers
                - bottom: dropping the bottom K layers
                - alternate: one layer is drop, the following is dropped, and so on.
                - symmetric: drops the K middle layers.
            K (int):
                number of layers to drop
        Returns:
            List[bool]: list of boolean indicating whether to drop layer i.
        """
        assert dropping_technique in [
            "top",
            "bottom",
            "alternate",
            "symmetric",
        ], f"Dropping technique '{dropping_technique} not supported."

        N = len(self.layer)
        if dropping_technique == "bottom":
            layer_to_drop = [True] * K + [False] * (N - K)
        elif dropping_technique == "top":
            layer_to_drop = [False] * (N - K) + [True] * K
        elif dropping_technique == "alternate":
            layer_to_drop = []
            for _ in range(K):
                layer_to_drop += [True, False]

            layer_to_drop += [False] * (N - 2 * K)
        else:
            layer_to_drop = [False] * N
            middle = N // 2
            layer_to_drop[middle - K // 2 : middle + K // 2] = [True] * K
        return layer_to_drop

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """"""
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.layer_to_drop and self.layer_to_drop[i]:
                continue

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                logging.info(f"Skipping layer {i}")
                continue

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logging.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CustomBertModel(BertModel, ABC):
    """
    Custom BERT model using `BertCustomEncoder` to perform layer dropping.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture.
        add_pooling_layer (bool):
            whether to add a pooling layer on top of the model.
    """

    def __init__(self, config: PretrainedConfig, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer)
        self.encoder = BertCustomEncoder(config)

    @classmethod
    def from_config(cls, config: PretrainedConfig, **kwargs):
        """
        Create a CustomBertModel object from a given config.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model
