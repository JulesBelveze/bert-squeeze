import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    """
    Kullback–Leibler divergence loss using a softmax temperature.

    Args:
        T (float):
            temperature parameter with default to 1.0, which is equivalent to a regular
            softmax. Increasing the temperature parameter will penalize bigger values,
            owing to the amplification effect of the exponential, which leads to decrease
            the model's confidence.
    """

    def __init__(self, T: float = 1.0):
        super(KLDivLoss, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss()

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor):
                predictions of the student model
            teacher_logits (torch.Tensor):
                predictions of the teacher model
        Returns:
            torch.Tensor:
                Kullback-Leibler divergence
        """
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        p_s = F.log_softmax(student_logits / self.T)
        return self.kl(p_s, p_t) * (self.T**2)
