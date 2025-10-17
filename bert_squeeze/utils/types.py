import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypeVar, Union

import torch


@dataclass
class RampOutput:
    logits: torch.Tensor
    pooled_output: torch.Tensor
    entropy: Optional[Union[float, torch.Tensor]] = None

    def __getitem__(self, item):
        """"""
        if self.logits.squeeze().dim() == 1:
            return self
        return RampOutput(
            logits=self.logits[item],
            pooled_output=self.pooled_output[item],
            entropy=None if self.entropy is None else self.entropy[item],
        )


@dataclass
class DeeBertEncoderOutput:
    exit_layer: int
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ramps_exit: Optional[Tuple[RampOutput]] = None
    # Optional per-layer gate logits/probs for BERxiT
    gates_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DeeBertModelOutput:
    exit_layer: int
    sequence_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[torch.FloatTensor] = None
    ramps_exits: Optional[torch.FloatTensor] = None
    # Optional per-layer gate logits/probs for BERxiT
    gates_logits: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def logits(self) -> torch.Tensor:
        """"""
        return torch.stack([ramp.logits for ramp in self.ramps_exits], dim=0)


@dataclass
class DistillationLoss:
    kd_loss: torch.Tensor
    objective: torch.Tensor
    full_loss: torch.Tensor


# TODO: find a way not to hardcode the number of layers
FastBertLoss = dataclasses.make_dataclass(
    cls_name="FastBertLoss",
    fields=[("full_loss", torch.Tensor)]
    + [(f"kl_layer_{i}", Optional[torch.Tensor], field(default=None)) for i in range(11)],
)

Loss = TypeVar("Loss", DistillationLoss, FastBertLoss, torch.Tensor)
LossType = Optional[Loss]
