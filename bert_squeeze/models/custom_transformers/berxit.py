from abc import ABC
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertPreTrainedModel,
)

from ...utils.losses import entropy
from ...utils.types import DeeBertEncoderOutput, DeeBertModelOutput, RampOutput
from .deebert import OffRamp


class BerxitEncoder(nn.Module):
    """BERT encoder with classifier ramps and a shared exit gate."""

    def __init__(self, config: PretrainedConfig, inference: bool):
        super(BerxitEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ramp = nn.ModuleList(
            [OffRamp(config) for _ in range(config.num_hidden_layers)]
        )
        self.gates = nn.Linear(config.hidden_size, 1)

        self.early_exit_entropy = [-1.0] * config.num_hidden_layers
        self.gate_thresholds = [-1.0] * config.num_hidden_layers
        self.inference = inference

    def set_early_exit_entropy(self, x: Union[List[float], float]) -> None:
        if isinstance(x, float) or isinstance(x, int):
            for i in range(self.config.num_hidden_layers):
                self.early_exit_entropy[i] = float(x)
        elif isinstance(x, list):
            self.early_exit_entropy = x
        else:
            raise TypeError(
                f"Expected 'x' to be of type 'float' or 'list' but got :'{type(x)}'"
            )

    def init_highway_pooler(self, pooler: torch.nn.ModuleDict) -> None:
        """Initialize each ramp pooler with the main pooler weights."""
        loaded_model = pooler.state_dict()
        for ramp in self.ramp:
            for name, param in ramp.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def set_exit_gate_thresholds(self, x: Union[List[float], float]) -> None:
        if isinstance(x, float) or isinstance(x, int):
            self.gate_thresholds = [float(x)] * self.config.num_hidden_layers
        elif isinstance(x, list):
            if len(x) != self.config.num_hidden_layers:
                raise ValueError("Gate threshold count must match the encoder depth.")
            self.gate_thresholds = [float(v) for v in x]
        else:
            raise TypeError(
                f"Expected 'x' to be of type 'float' or 'list' but got :'{type(x)}'"
            )

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
        all_hidden_states: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        if not self.inference:
            all_ramps: List[RampOutput] = []
            all_gates: List[torch.Tensor] = []
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    attention = layer_outputs[1]
                    all_attentions.append(attention)

                ramp_exit = self.ramp[i](hidden_states)
                all_ramps.append(ramp_exit)
                gate_logit = self.gates(hidden_states[:, 0])
                all_gates.append(gate_logit)

                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

            return DeeBertEncoderOutput(
                last_hidden_state=hidden_states,
                hidden_states=(
                    tuple(all_hidden_states) if output_hidden_states else None
                ),
                attentions=tuple(all_attentions) if output_attentions else None,
                ramps_exit=tuple(all_ramps),
                gates_logits=tuple(all_gates),
                exit_layer=i,
            )

        batch_size = hidden_states.shape[0]
        inference_ramps: List[Optional[RampOutput]] = [None] * batch_size
        positions = torch.arange(
            start=0, end=hidden_states.shape[0], device=hidden_states.device
        ).long()
        gates_per_layer: Tuple[torch.Tensor, ...] = tuple(
            torch.full(
                (batch_size, 1),
                float("nan"),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            for _ in range(len(self.layer))
        )

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            ramp_exit = self.ramp[i](hidden_states)
            gate_logit = self.gates(hidden_states[:, 0])
            gates_per_layer[i][positions] = gate_logit
            gate_prob = torch.sigmoid(gate_logit).squeeze(-1)
            ramp_entropy = entropy(ramp_exit.logits)
            ramp_exit.entropy = ramp_entropy

            is_final_layer = i == len(self.layer) - 1
            if is_final_layer:
                enough_info = torch.ones_like(gate_prob, dtype=torch.bool)
            elif self.gate_thresholds[i] >= 0:
                enough_info = gate_prob >= self.gate_thresholds[i]
            else:
                enough_info = ramp_entropy < self.early_exit_entropy[i]
            right_pos = positions[enough_info]

            for idx, pos in enumerate(right_pos):
                inference_ramps[pos] = ramp_exit[idx]

            if is_final_layer:
                continue

            hidden_states = hidden_states[~enough_info]
            attention_mask = attention_mask[~enough_info]
            positions = positions[~enough_info]

            if positions.nelement() == 0:
                break

        completed_ramps = tuple(ramp for ramp in inference_ramps if ramp is not None)
        if len(completed_ramps) != batch_size:
            raise RuntimeError("BERxiT inference did not produce every sample.")
        return DeeBertEncoderOutput(
            ramps_exit=completed_ramps,
            gates_logits=gates_per_layer,
            exit_layer=i,
        )


class BerxitModel(BertPreTrainedModel, ABC):
    """BERT model with BERxiT early exits."""

    def __init__(self, config: PretrainedConfig, inference: bool = False):
        super(BerxitModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BerxitEncoder(config, inference)
        self.pooler = BertPooler(config)

        self.init_weights()

    def set_inference_mode(self, inference: bool) -> None:
        self.encoder.inference = inference

    def init_highway_pooler(self) -> None:
        self.encoder.init_highway_pooler(self.pooler)

    def set_exit_gate_thresholds(self, x: Union[List[float], float]) -> None:
        self.encoder.set_exit_gate_thresholds(x)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune) -> None:
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
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> DeeBertModelOutput:
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

        # Broadcast attention masks
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
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

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask
        ) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = (
            None if self.encoder.inference else encoder_outputs.last_hidden_state
        )
        pooled_output = None if self.encoder.inference else self.pooler(sequence_output)

        return DeeBertModelOutput(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            ramps_exits=encoder_outputs.ramps_exit,
            gates_logits=encoder_outputs.gates_logits,
            exit_layer=encoder_outputs.exit_layer,
        )
