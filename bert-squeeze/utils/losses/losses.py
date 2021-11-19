import numpy as np
import torch


def normal_shannon_entropy(p: torch.Tensor, num_labels: int) -> float:
    entrop = torch.distributions.Categorical(probs=p).entropy()
    normal = -np.log(1.0 / num_labels)
    return entrop / normal


def entropy(x: torch.Tensor) -> float:
    """
    :param x: logits before softmax
    :return: entropy
    """
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B / A
