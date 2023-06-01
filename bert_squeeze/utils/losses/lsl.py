import torch


class LabelSmoothingLoss(torch.nn.Module):
    """
    Label Smoothing Cross Entropy loss.

    This loss aims at penalize overconfident predictions and improve generalization. Label
    smoothing can be seen as distribution regularization.

    Args:
        nb_classes (int):
            number of different classes
        smoothing (float):
            smoothing factor, that needs to be in [0,1]. If `smoothing=0` we obtain the original
            Cross Entropy loss and if `smoothing=1` we get a uniform distribution
    """

    def __init__(self, nb_classes: int, smoothing: float = 0.0, dim: int = -1):
        super(LabelSmoothingLoss, self).__init__()
        assert (
            0 <= smoothing < 1
        ), f"Smoothing parameter should be between 0 and 1, got {smoothing}"

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.nb_classes = nb_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor):
                predicted logits
            target (torch.Tensor):
                ground truth labels

        Returns:
            torch.Tensor:
                Label smoothed cross entropy loss
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.nb_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
