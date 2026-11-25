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


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor):
            tensor of unnormalized predicted logits, one row per sample
    Returns:
        torch.Tensor:
            entropy of the predicted distribution of each sample
    """
    # Always normalize with a softmax. Guessing between probabilities and logits from
    # the values made the result depend on whether the batch held a negative logit.
    return torch.distributions.Categorical(logits=logits).entropy()
