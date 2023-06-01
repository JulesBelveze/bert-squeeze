from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RomeBertLossOutput:
    full_loss: torch.Tensor
    distill_loss: torch.Tensor
    multi_loss: torch.Tensor
    last_loss: torch.Tensor


class RomeBertLoss(nn.Module):
    """
    RomeBert self-distillation loss used in the paper (see https://arxiv.org/pdf/2101.09755.pdf for
    more information)

    Disclaimer: this is heavily inspired by the repo's implementation
    """

    def __init__(self, n_blocks, gamma, T, num_labels):
        super(RomeBertLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.n_blocks = n_blocks
        self.gamma = gamma
        self.T = T
        self.num_labels = num_labels

    def forward(
        self,
        logits: torch.Tensor,
        highway_logits: torch.Tensor,
        targets: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> RomeBertLossOutput:
        """
        Args:
            logits (torch.Tensor):

            highway_logits (torch.Tensor):
            targets (torch.Tensor):
            soft_targets (torch.Tensor):

        Returns:

        """
        if self.num_labels == 1:
            last_loss = self.mse_loss(logits.view(-1), targets.view(-1))
        else:
            last_loss = self.ce_loss(logits.view(-1, self.num_labels), targets.view(-1))

        T = self.T
        multi_losses = []
        distill_losses = []
        for i in range(self.n_blocks - 1):
            if self.num_labels == 1:
                _mse = (1.0 - self.gamma) * self.mse_loss(
                    highway_logits[i].view(-1), targets.view(-1)
                )
                _kld = (
                    self.kld_loss(
                        self.log_softmax(highway_logits[i].view(-1) / T),
                        self.softmax(soft_targets.view(-1) / T),
                    )
                    * self.gamma
                    * T
                    * T
                )
                multi_losses.append(_mse)
                distill_losses.append(_kld)
            else:
                _ce = (1.0 - self.gamma) * self.ce_loss(
                    highway_logits[i].view(-1, self.num_labels), targets.view(-1)
                )
                _kld = (
                    self.kld_loss(
                        self.log_softmax(highway_logits[i].view(-1, self.num_labels) / T),
                        self.softmax(soft_targets.view(-1, self.num_labels) / T),
                    )
                    * self.gamma
                    * T
                    * T
                )
                multi_losses.append(_ce)
                distill_losses.append(_kld)

        m_loss = sum(multi_losses)
        d_loss = sum(distill_losses)
        l_loss = last_loss
        loss = l_loss + d_loss + m_loss

        return RomeBertLossOutput(
            full_loss=loss, distill_loss=d_loss, multi_loss=m_loss, last_loss=l_loss
        )
