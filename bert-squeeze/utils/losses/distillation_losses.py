import torch.nn as nn
import torch.nn.functional as  F
from ..types import KDLossOutput


class KLDivLoss(nn.Module):
    def __init__(self, T: float = 1.0):
        super(KLDivLoss, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss()

    def forward(self, student_logits, teacher_logits):
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        p_s = F.log_softmax(student_logits / self.T)
        return self.kl(p_s, p_t) * (self.T ** 2)


class KDLoss(nn.Module):
    """
    Copied from:
    github.com/romebert/RomeBERT/blob/b4eb26bfcf202f398aa2c699c08b7234058a3261/transformers/new_modeling_highway_bert.py
    """

    def __init__(self, n_blocks, gamma, T, num_labels):
        super(KDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss().cuda()
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.n_blocks = n_blocks
        self.gamma = gamma
        self.T = T
        self.num_labels = num_labels

    def forward(self, outputs, highway_outputs, targets, soft_targets):
        if self.num_labels == 1:
            last_loss = self.mse_loss(outputs.view(-1), targets.view(-1))
        else:
            last_loss = self.ce_loss(outputs.view(-1, self.num_labels), targets.view(-1))

        T = self.T
        multi_losses = []
        distill_losses = []
        for i in range(self.n_blocks - 1):
            if self.num_labels == 1:
                _mse = (1. - self.gamma) * self.mse_loss(highway_outputs[i].view(-1), targets.view(-1))
                _kld = self.kld_loss(self.log_softmax(highway_outputs[i].view(-1) / T),
                                     self.softmax(soft_targets.view(-1) / T)) * self.gamma * T * T
                multi_losses.append(_mse)
                distill_losses.append(_kld)
            else:
                _ce = (1. - self.gamma) * self.ce_loss(highway_outputs[i].view(-1, self.num_labels), targets.view(-1))
                _kld = self.kld_loss(self.log_softmax(highway_outputs[i].view(-1, self.num_labels) / T),
                                     self.softmax(soft_targets.view(-1, self.num_labels) / T)) * self.gamma * T * T
                multi_losses.append(_ce)
                distill_losses.append(_kld)

        m_loss = sum(multi_losses)
        d_loss = sum(distill_losses)
        l_loss = last_loss
        loss = l_loss + d_loss + m_loss

        return KDLossOutput(
            full_loss=loss,
            distill_loss=d_loss,
            multi_loss=m_loss,
            last_loss=l_loss
        )
