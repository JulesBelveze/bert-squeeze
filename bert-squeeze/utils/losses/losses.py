import torch


def entropy(x: torch.Tensor) -> float:
    """
    :param x: logits before softmax
    :return: entropy
    """
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B / A
