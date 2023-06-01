# This piece of code is heavily insipred by:
# https://github.com/JetRunner/BERT-of-Theseus/blob/master/bert_of_theseus/modeling_bert_of_theseus.py

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from abc import ABC
from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertPreTrainedModel,
)

logger = logging.getLogger(__name__)


class TheseusBertEncoder(nn.Module):
    """
    Implementation of the TheseusBert encoder presented in the paper: "BERT-of-Theseus:
    Compressing BERT by Progressive Module Replacing" (https://arxiv.org/pdf/2002.02925.pdf).

    The general idea behind is to be able to compress the original BERT model by dividing
    it into submodules which will be substituted by a compressed version (aka "successor_layers").

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture.
        nb_successor_layers (int):
            total number of layers in the compressed model.
    """

    def __init__(self, config: PretrainedConfig, nb_successor_layers: int = 6):
        super(TheseusBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.nb_predecessor_layer = config.num_hidden_layers
        self.nb_successor_layers = nb_successor_layers
        assert (
            self.nb_predecessor_layer % self.nb_successor_layers == 0
        ), "The number of predecessor layers must be a multiple of the number of sucessor layers."
        self.compress_ratio = self.nb_predecessor_layer // self.nb_successor_layers

        self.bernoulli = None

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.nb_predecessor_layer)]
        )
        self.successor_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.nb_successor_layers)]
        )

    def init_successor_layers(self) -> None:
        """
        Creates the successor layers by copying layers from the original model.
        """
        self.successor_layers = nn.ModuleList(
            [deepcopy(self.layer[ix]) for ix in range(self.nb_successor_layers)]
        )

    def set_replacing_rate(self, replacing_rate: float) -> None:
        """
        Sets the probability to use in the Bernoulli distribution which will later
        be used to compute the model's outputs.

        Args:
            replacing_rate (float):
                probability to use in the Bernoulli distribution
        """
        if not 0.0 < replacing_rate <= 1.0:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        During the training for every successor layer we sample the Bernoulli distribution and
        replace the original layer with its corresponding successor layer  with probability
        `replacing_rate` and keep the original with probability `1 - replacing_rate`.

        At inference time all the original layers are replaced by their successor layers.

        Args:
            hidden_states:
            attention_mask:
            head_mask:
            encoder_hidden_states:
            encoder_attention_mask:

        Returns:

        """
        all_hidden_states = ()
        all_attentions = ()
        if self.training:
            inference_layers = []
            for i in range(self.nb_successor_layers):
                if self.bernoulli.sample() == 1:  # replace
                    inference_layers.append(self.successor_layers[i])
                else:  # keep
                    for offset in range(self.compress_ratio):
                        inference_layers.append(
                            self.layer[i * self.compress_ratio + offset]
                        )

        else:  # inference with compressed model
            inference_layers = self.successor_layers

        for i, layer_module in enumerate(inference_layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )


class TheseusBertModel(BertPreTrainedModel, ABC):
    """
    Implementation of the TheseusBert presented in the paper: "BERT-of-Theseus: Compressing
    BERT by Progressive Module Replacing" (https://arxiv.org/pdf/2002.02925.pdf).

    The implementation is similar to the `transformers.BertModel` one except that it uses
    the previously defined `TheseusBertEncoder` as encoder.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture.
    """

    def __init__(self, config: PretrainedConfig):
        super(TheseusBertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = TheseusBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (Dict[int, List[int]]):
                dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """"""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                causal_mask = causal_mask.to(
                    torch.long
                )  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if needed + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
