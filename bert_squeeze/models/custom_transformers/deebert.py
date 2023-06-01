# This is heavily inspired by the following repo:
# https://github.com/castorini/DeeBERT
from abc import ABC
from typing import List, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertPreTrainedModel,
)

from ...utils.errors import RampException
from ...utils.losses import entropy
from ...utils.types import DeeBertEncoderOutput, DeeBertModelOutput, RampOutput


class OffRamp(nn.Module):
    """
    Classification layers (referred to as off-ramps) in the DeeBERT paper.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture
    """

    def __init__(self, config: PretrainedConfig):
        super(OffRamp, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> RampOutput:
        """
        Args:
            hidden_states (torch.Tensor):
                output of the previous transformer layer.
        Returns:
            RampOutput: output of the given off-ramp containing both the classification logits
            and the pooled outputs.
        """
        # accessing the first token
        pooled_output = self.pooler(hidden_states[:, 0].unsqueeze(1))
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return RampOutput(logits=logits, pooled_output=pooled_output)


class DeeBertEncoder(nn.Module):
    """
    We construct the DeeBertEncoder as a regular BertEncoder except that we squeeze
    an OffRamp (a classification layer) between each BertLayer.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture
    """

    def __init__(self, config: PretrainedConfig):
        super(DeeBertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config)] * config.num_hidden_layers)
        self.ramp = nn.ModuleList([OffRamp(config)] * config.num_hidden_layers)

        self.early_exit_entropy = [
            -1,
        ] * config.num_hidden_layers

    def set_early_exit_entropy(self, x: Union[List[float], float]) -> None:
        """
        Assigning an entropy threshold to every layer.

        Args:
            x (Union[List[float], float]):
                entropy threshold to assign to the layer, can either be a float that will be
                set to all layers or a list of different thresholds to set to individual layers.
        """
        if isinstance(x, float) or isinstance(x, int):
            for i in range(self.config.num_hidden_layers):
                self.early_exit_entropy[i] = x
        elif isinstance(x, list):
            self.early_exit_entropy = x
        else:
            raise TypeError(
                f"Expected 'x' to be of type 'float' or 'list' but got :'{type(x)}'"
            )

    def init_highway_pooler(self, pooler: torch.nn.ModuleDict) -> None:
        """
        Initialize the pooling layer of each ramp based on a given pooling layer.

        Args:
            pooler (torch.nn.ModuleDict):
                holds modules in a dictionary.
        """
        loaded_model = pooler.state_dict()
        for ramp in self.ramp:
            for name, param in ramp.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> DeeBertEncoderOutput:
        """"""
        all_hidden_states = tuple() if output_hidden_states else None
        all_attentions = tuple() if output_attentions else None
        all_ramps = tuple()

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )  # output of the BertLayer
            hidden_states = layer_outputs[0]  # (bs * seq_len * hidden_dim)

            if output_attentions:
                attention = layer_outputs[1]
                all_attentions += (attention,)

            ramp_exit = self.ramp[i](hidden_states)  # RampOutput

            # At test time we sequentially compute the entropy of the output of each ramp
            # and exit the model by throwing an exception when enough information has been
            # computed.
            # During training, we want to train all the layers regardless the entropy.
            if not self.training:
                ramp_exit.entropy = entropy(ramp_exit.logits)
                all_ramps += (ramp_exit,)

                # we don't go deeper in the model if the entropy is lower than the set threshold
                if ramp_exit.entropy < self.early_exit_entropy[i]:
                    raise RampException(all_ramps, i + 1)
            else:
                all_ramps += (ramp_exit,)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return DeeBertEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            ramps_exit=all_ramps,
        )


class DeeBertModel(BertPreTrainedModel, ABC):
    """
    Bare DeeBert model without head on top, outputting hidden states.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture
    """

    def __init__(self, config: PretrainedConfig):
        super(DeeBertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = DeeBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def init_highway_pooler(self) -> None:
        """
        Initializes the pooler layer from each ramp of the encoder with
        its current pooler layer
        """
        self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune) -> None:
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
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
    ) -> DeeBertModelOutput:
        """ """
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
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

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
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask
        ) * -10000.0

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
            )  # switch to fload if need + fp16 compatibility
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

        return DeeBertModelOutput(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            hidden_states=encoder_hidden_states,
            attentions=encoder_outputs.attentions,
            ramps_exits=encoder_outputs.ramps_exit,
        )
