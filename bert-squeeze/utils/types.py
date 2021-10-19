import torch
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import TypeVar, Optional


class DeeBertEncoderOutput(ModelOutput):
    # TODO: check type
    last_hidden_state: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    ramps_exit: torch.FloatTensor = None


class DeeBertModelOutput(ModelOutput):
    sequence_output: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    ramps_exits: torch.FloatTensor = None


class RomeBertEncoderOutput(ModelOutput):
    # TODO: check type
    last_hidden_state: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    ramps_exit: torch.FloatTensor = None


class RomeBertModelOutput(ModelOutput):
    sequence_output: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    ramps_exits: torch.FloatTensor = None


@dataclass
class DistillationLoss:
    kd_loss: torch.Tensor
    objective: torch.Tensor
    full_loss: torch.Tensor


@dataclass
class KDLossOutput:
    full_loss: torch.Tensor
    distill_loss: torch.Tensor
    multi_loss: torch.Tensor
    last_loss: torch.Tensor


Loss = TypeVar("Loss", KDLossOutput, DistillationLoss)
LossType = Optional[Loss]
