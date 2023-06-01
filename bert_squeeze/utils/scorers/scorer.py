from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from overrides import overrides
from tabulate import tabulate

from ..types import DistillationLoss, LossType


class Scorer:
    """
    Helper class to compute various evaluation metrics.

    This is achieved by updating a confusion matrix after every batch.
    """

    def __init__(self, n_labels: int):
        """
        Args:
            n_labels (int):
                number of available labels
        """
        self.n_labels = n_labels
        self.confusion_matrix = np.zeros((n_labels, n_labels))
        self.batch_counter = 0
        self.eps = 1e-8
        self.losses = defaultdict(list)

    @property
    def acc(self) -> float:
        """
        Computes the accuracy score:
            number of correct predictions / number of predictions

        Returns:
            float: accuracy score
        """
        return self.confusion_matrix.trace() / (self.confusion_matrix.sum() + self.eps)

    @property
    def precision(self) -> np.array:
        """
        Computes the precision score for each class using the following formula:
            tp / (tp + fp)

        Returns:
            np.array: precision score for every class. precision[i] is the precision score
                      of class i
        """
        return self.confusion_matrix.diagonal() / (
            self.confusion_matrix.sum(axis=-1) + self.eps
        )

    @property
    def macro_precision(self) -> ndarray:
        """
        Computes the unweighted mean of the precisions

        Returns:
            float: unweighted mean of the precisions
        """
        return np.mean(self.precision)

    @property
    def micro_precision(self) -> float:
        """
        Counts the total true positives, false negatives and false positives.

        Returns:
            float: micro precision
        """
        return self.acc

    @property
    def weighted_precision(self) -> float:
        """
        Computes the precision for each label, and find their average weighted by support.

        Returns:
            float: average weighted precision
        """
        return (self.confusion_matrix.sum(axis=0) * self.precision) / (
            self.confusion_matrix.sum() + self.eps
        )

    @property
    def recall(self) -> np.array:
        """
        Computes the recall score for each class using the following formula:
            tp / (tp + fn)

        Returns:
            np.array: recall score for every class. recall[i] is the recall score
                      of class i
        """
        return self.confusion_matrix.diagonal() / (
            self.confusion_matrix.sum(axis=0) + self.eps
        )

    @property
    def macro_recall(self) -> ndarray:
        """
        Computes the unweighted mean of the recalls

        Returns:
            float: unweighted mean of the recalls
        """
        return np.mean(self.recall)

    @property
    def micro_recall(self) -> float:
        """
        Counts the total true positives, false negatives and false positives.

        Returns:
            float: micro recall
        """
        return self.acc

    @property
    def weighted_recall(self) -> float:
        """
        Computes the recall for each label, and find their average weighted by support.

        Returns:
            float: average weighted recall
        """
        return (self.confusion_matrix.sum(axis=0) * self.recall) / (
            self.confusion_matrix.sum() + self.eps
        )

    @property
    def f1(self) -> np.array:
        """
        Computes the f1 score for each class using the following formula:
            F1 = 2 * (precision * recall) / (precision + recall)

        Returns:
            np.array: f1 score for every class. f1[i] is the f1 score of class i
        """
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall + self.eps)
        )

    @property
    def macro_f1(self) -> ndarray:
        """
        Computes the unweighted mean of the f1s

        Returns:
            float: unweighted mean of the f1s
        """
        return np.mean(self.f1)

    @property
    def micro_f1(self) -> float:
        """
        Counts the total true positives, false negatives and false positives.

        Returns:
            float: micro f1
        """
        return self.acc

    @property
    def weighted_f1(self) -> float:
        """
        Computes the f1 for each label, and find their average weighted by support.

        Returns:
            float: average weighted f1
        """
        return (self.confusion_matrix.sum(axis=0) * self.f1) / (
            self.confusion_matrix.sum() + self.eps
        )

    def _inc_counter(self) -> None:
        """
        Increments the batch counter
        """
        self.batch_counter += 1

    def add(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: Union[LossType, float] = None,
        ignored_label: int = -100,
    ) -> None:
        """
        Updates the confusion matrix by using the predictions and the ground truth labels.

        Args:
            logits (torch.Tensor):
                predicted logits
            labels (torch.Tensor):
                ground truth labels
            loss (Union[LossType, float]):
                computed loss
            ignored_label (int):
                value to ignore during metric computation, default to -100 (PyTorch ignored value)
        """
        self._inc_counter()

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        preds_np = preds.flatten().detach().cpu().numpy()
        labels_np = labels.flatten().detach().cpu().numpy()

        for y, y_hat in zip(labels_np, preds_np):
            if y != ignored_label:
                self.confusion_matrix[y][y_hat] += 1

        if loss is None:
            return
        elif isinstance(loss, float) or isinstance(loss, torch.Tensor):
            self.losses["global"].append(loss)
        elif isinstance(loss, DistillationLoss):
            for attr in loss.__dataclass_fields__:
                self.losses[attr].append(getattr(loss, attr).detach())
        else:
            raise TypeError(f"Got 'loss' with type: {type(loss)}")

    def reset(self) -> None:
        """
        Resets the confusion matrix, batch counter and losses.
        This method should be called after logging metrics.
        """
        self.batch_counter = 0
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.losses = defaultdict(list)

    def to_dict(self) -> Dict[str, float]:
        """
        Returns all the accessible metrics within a dict where the key is the metric name
        and the value is the metric.

        Returns:
            Dict[str, float]: dict of metrics
        """
        return {
            "acc": self.acc,
            "p": self.precision,
            "micro-p": self.micro_precision,
            "macro-p": self.macro_precision,
            "weighted-p": self.weighted_precision,
            "r": self.recall,
            "micro-r": self.micro_recall,
            "macro-r": self.macro_recall,
            "weighted-r": self.weighted_recall,
            "f1": self.f1,
            "micro-f1": self.micro_f1,
            "macro-f1": self.macro_f1,
            "weighted-f1": self.weighted_f1,
        }

    def get_table(self) -> str:
        """
        Method to format all the metrics into a pretty table.

        Returns:
            str: prettyfied table summarizing all the metrics
        """
        table = []
        for k, v in self.to_dict().items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.float64):
                v = [v.item()]
            table.append([k] + v)
        return tabulate(
            table,
            headers=["metrics"] + [f"class {i}" for i in range(self.n_labels)],
            tablefmt="fancy_grid",
        )


class LooseScorer(Scorer):
    """
    Helper class to compute various 'loose' evaluation metrics.

    For some classification one might be interested in gathering some classes.

    E.g. an anomaly detection problem with different severity level:
    0: not an anomaly, 1: mild anomaly, 2: anomaly, 3: severe anomaly
    One could be interested in checking the performance of the loose
    problem: anomaly/not anomaly. To do so:
    >>> loose_scorer = LooseScorer(loose_classes=[[0], [1,2,3]])
    """

    def __init__(self, loose_classes: List[List[Union[str, int]]]):
        """
        Args:
            loose_classes (List[List[Union[str, int]]]):
                list of classes to aggregate together during metrics computation
        """
        super().__init__(n_labels=2)
        self.loose_classes = loose_classes

    @overrides
    def add(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: Union[LossType, float] = None,
        ignored_label: int = -100,
    ) -> None:
        """
        Updates the confusion matrix by using the predictions and the ground truth labels and keeps
        track of the loss.

        Args:
            logits (torch.Tensor):
                predicted logits
            labels (torch.Tensor):
                ground truth labels
            loss (Union[LossType, float]):
                computed loss
            ignored_label (int):
                value to ignore during metric computation, default to -100 (PyTorch ignored value)
        """
        self._inc_counter()

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).flatten()
        labels = labels.flatten()

        bin_preds = [1 if elt in self.loose_classes[1] else 0 for elt in preds]
        bin_labels = [1 if elt in self.loose_classes[1] else 0 for elt in labels]

        for y, y_hat in zip(bin_labels, bin_preds):
            if y != ignored_label:
                self.confusion_matrix[y][y_hat] += 1

        if loss is None:
            return
        elif isinstance(loss, float) or isinstance(loss, torch.Tensor):
            self.losses["global"].append(loss)
        elif isinstance(loss, LossType):
            for attr in loss.__dataclass_fields__:
                self.losses[attr].append(getattr(loss, attr).detach())
        else:
            raise TypeError(f"Got 'loss' with type: {type(loss)}")
