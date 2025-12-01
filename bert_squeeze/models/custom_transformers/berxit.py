"""
This implementation mirrors the DeeBERT integration but under the BERxiT
name, following the repository's integration patterns so users can train
and use a BERxiT-style early-exiting model.

Note: This module reuses the same architectural approach as DeeBERT in this
codebase (off-ramps between layers with entropy-based early exit),
so it integrates seamlessly with existing training loops and configs.
"""

from abc import ABC
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertPreTrainedModel,
)

from ...utils.losses import entropy
from ...utils.types import DeeBertEncoderOutput, DeeBertModelOutput, RampOutput


class BerxitOffRamp(nn.Module):
    """
    Classification layers (off-ramps) placed between encoder layers.

    Args:
        config (PretrainedConfig): configuration defining the model architecture
    """

    def __init__(self, config: PretrainedConfig):
        super(BerxitOffRamp, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> RampOutput:
        # Access the first token and apply pooler similar to BERT's default
        pooled_output = self.pooler(hidden_states[:, 0].unsqueeze(1))
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return RampOutput(logits=logits, pooled_output=pooled_output)


class ExitGate(nn.Module):
    """A small MLP gate that predicts whether to exit at a given layer.

    Inputs are hand-crafted features from the ramp logits/probs.
    """

    def __init__(self, in_dim: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Returns logits for BCEWithLogitsLoss
        return self.net(feats)


class BerxitEncoder(nn.Module):
    """
    Encoder that inserts off-ramps between each Transformer block and
    supports early exiting using an entropy threshold.
    """

    def __init__(self, config: PretrainedConfig, inference: bool):
        super(BerxitEncoder, self).__init__()
        self.config = config
        # Ensure distinct modules per layer
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.ramp = nn.ModuleList([BerxitOffRamp(config) for _ in range(config.num_hidden_layers)])
        # BERxiT gates
        gate_hidden = getattr(config, "gate_hidden_dim", 32)
        self.gates = nn.ModuleList([ExitGate(in_dim=3, hidden=gate_hidden) for _ in range(config.num_hidden_layers)])

        # Thresholds for DeeBERT entropy (fallback) and BERxiT gate
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
            assert len(x) == self.config.num_hidden_layers, "gate thresholds size mismatch"
            self.gate_thresholds = [float(v) for v in x]
        else:
            raise TypeError(
                f"Expected 'x' to be of type 'float' or 'list' but got :'{type(x)}'"
            )

    @staticmethod
    def _gate_features(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        probs = F.softmax(logits, dim=-1)
        pmax, _ = probs.max(dim=-1, keepdim=True)  # [B,1]
        top2 = torch.topk(probs, k=2, dim=-1).values  # [B,2]
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)  # [B,1]
        ent = entropy(probs).unsqueeze(-1)  # [B,1]
        return torch.cat([pmax, margin, ent], dim=-1)  # [B,3]

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
        all_hidden_states = tuple() if output_hidden_states else None
        all_attentions = tuple() if output_attentions else None

        if not self.inference:
            all_ramps: Tuple[RampOutput, ...] = tuple()
            all_gates: Tuple[torch.Tensor, ...] = tuple()
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

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
                    all_attentions += (attention,)

                ramp_exit = self.ramp[i](hidden_states)
                all_ramps += (ramp_exit,)
                # Gate logits from features of current ramp
                feats = self._gate_features(ramp_exit.logits)
                gate_logit = self.gates[i](feats)  # [B,1]
                all_gates += (gate_logit,)

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            return DeeBertEncoderOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                ramps_exit=all_ramps,
                gates_logits=all_gates,
                exit_layer=i,
            )
        else:
            batch_size = hidden_states.shape[0]
            all_ramps = [0] * batch_size
            positions = torch.arange(
                start=0, end=hidden_states.shape[0], device=hidden_states.device
            ).long()
            # Collect per-layer gate logits for diagnostics; fill with NaNs by default
            gates_per_layer: Tuple[torch.Tensor, ...] = tuple(
                torch.full((batch_size, 1), float('nan'), device=hidden_states.device, dtype=hidden_states.dtype)
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
                # Compute gate decision
                feats = self._gate_features(ramp_exit.logits)
                gate_logit = self.gates[i](feats)
                # Scatter current gate logits back to original batch positions
                gates_per_layer[i][positions] = gate_logit
                gate_prob = torch.sigmoid(gate_logit).squeeze(-1)  # [b_cur]
                ramp_exit.entropy = entropy(ramp_exit.logits)

                if i == len(self.layer) - 1:
                    for idx, pos in enumerate(positions):
                        all_ramps[pos] = ramp_exit[idx]
                else:
                    # Prefer gate thresholds; fallback to entropy if thresholds are negative
                    if self.gate_thresholds[i] >= 0:
                        enough_info = gate_prob >= self.gate_thresholds[i]
                    else:
                        enough_info = ramp_exit.entropy < self.early_exit_entropy[i]
                    right_pos = positions[enough_info]

                    for idx, pos in enumerate(right_pos):
                        all_ramps[pos] = ramp_exit[idx]

                    hidden_states = hidden_states[~enough_info]
                    attention_mask = attention_mask[~enough_info]
                    positions = positions[~enough_info]

                    if positions.nelement() == 0:
                        return DeeBertEncoderOutput(
                            ramps_exit=all_ramps,
                            gates_logits=gates_per_layer,
                            exit_layer=i,
                        )
            return DeeBertEncoderOutput(
                ramps_exit=all_ramps,
                gates_logits=gates_per_layer,
                exit_layer=i,
            )


class BerxitModel(BertPreTrainedModel, ABC):
    """
    BERxiT-like BERT model with off-ramps and early exit, matching the
    interfaces used by the DeeBERT integration in this repository.
    """

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
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
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
        sequence_output = None if self.encoder.inference else encoder_outputs.last_hidden_state
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
