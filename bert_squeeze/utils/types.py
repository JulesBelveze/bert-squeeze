import dataclasses
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, TypeVar


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
    ramps_exit: Tuple[RampOutput] = None


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


# TODO: find a way not to hardcode the number of layers
FastBertLoss = dataclasses.make_dataclass(
    cls_name="FastBertLoss",
    fields=[("full_loss", torch.Tensor)] + [(f"kl_layer_{i}", Optional[torch.Tensor], field(default=None)) for i in
                                            range(11)]
)

Loss = TypeVar("Loss", DistillationLoss, FastBertLoss, torch.Tensor)
LossType = Optional[Loss]
