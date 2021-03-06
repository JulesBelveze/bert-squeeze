import dataclasses
from dataclasses import dataclass, field
from typing import TypeVar, Optional, Tuple

import torch


@dataclass
class RampOutput:
    logits: torch.Tensor
    pooled_output: torch.Tensor
    entropy: Optional[float] = None


@dataclass
class DeeBertEncoderOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ramps_exit: Tuple[torch.FloatTensor] = None


@dataclass
class DeeBertModelOutput:
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


# TODO: find a way not to hardcode the number of layers
FastBertLoss = dataclasses.make_dataclass(
    cls_name="FastBertLoss",
    fields=[("full_loss", torch.Tensor)] + [(f"kl_layer_{i}", Optional[torch.Tensor], field(default=None)) for i in
                                            range(11)]
)

Loss = TypeVar("Loss", KDLossOutput, DistillationLoss, FastBertLoss)
LossType = Optional[Loss]
