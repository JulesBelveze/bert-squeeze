import numpy as np
import torch


def normal_shannon_entropy(p: torch.Tensor, num_labels: int) -> torch.Tensor:
    """
    Computes the normalized Shannon Entropy (aka efficiency).
    See here: https://en.wikipedia.org/wiki/Entropy_(information_theory)#Efficiency_(normalized_entropy)

    Args:
        p (torch.Tensor):
            predicted probabilities
        num_labels (int):
            number of labels

    Returns:
        torch.Tensor:
            normalized Shannon entropy
    """
    entrop = torch.distributions.Categorical(probs=p).entropy()
    normal = -np.log(1.0 / num_labels)
    return entrop / normal


def entropy(p: torch.Tensor) -> torch.Tensor:
    """

    Args:
        p (torch.Tensor):
            tensor of predicted probabilities
    Returns:
        torch.Tensor:
            entropy of the predicted probabilities
    """
    try:
        return torch.distributions.Categorical(probs=p).entropy()
    except ValueError:
        return torch.distributions.Categorical(logits=p).entropy()
