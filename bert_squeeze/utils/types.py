from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

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
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    ramps_exit: Optional[Tuple[RampOutput, ...]] = None
    gates_logits: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class DeeBertModelOutput:
    exit_layer: int
    sequence_output: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    ramps_exits: Optional[Sequence[RampOutput]] = None
    gates_logits: Optional[Tuple[torch.Tensor, ...]] = None

    @property
    def logits(self) -> torch.Tensor:
        """"""
        if self.ramps_exits is None:
            raise ValueError("Ramp outputs are required to build logits.")
        return torch.stack([ramp.logits for ramp in self.ramps_exits], dim=0)


@dataclass
class SequenceClassificationOutput:
    logits: torch.Tensor
    intermediate_logits: Optional[Sequence[torch.Tensor]] = None
    ramps_exits: Optional[Sequence[RampOutput]] = None
    gates_logits: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    exit_layer: Optional[int] = None

    @property
    def scorer_logits(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.intermediate_logits is None:
            return self.logits
        return [*self.intermediate_logits, self.logits]


@dataclass
class DistillationLoss:
    kd_loss: torch.Tensor
    objective: torch.Tensor
    full_loss: torch.Tensor


@dataclass
class FastBertLoss:
    full_loss: torch.Tensor
    kl_layer_0: Optional[torch.Tensor] = None
    kl_layer_1: Optional[torch.Tensor] = None
    kl_layer_2: Optional[torch.Tensor] = None
    kl_layer_3: Optional[torch.Tensor] = None
    kl_layer_4: Optional[torch.Tensor] = None
    kl_layer_5: Optional[torch.Tensor] = None
    kl_layer_6: Optional[torch.Tensor] = None
    kl_layer_7: Optional[torch.Tensor] = None
    kl_layer_8: Optional[torch.Tensor] = None
    kl_layer_9: Optional[torch.Tensor] = None
    kl_layer_10: Optional[torch.Tensor] = None


@dataclass
class SequenceClassificationStepOutput:
    output: SequenceClassificationOutput
    labels: torch.Tensor
    loss: Union[torch.Tensor, FastBertLoss]

    @property
    def optimization_loss(self) -> torch.Tensor:
        if isinstance(self.loss, torch.Tensor):
            return self.loss
        return self.loss.full_loss


Loss = TypeVar("Loss", DistillationLoss, FastBertLoss, torch.Tensor)
LossType = Optional[Loss]
