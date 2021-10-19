from collections import defaultdict
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from overrides import overrides
from tabulate import tabulate

from .types import DistillationLoss, KDLossOutput, LossType


class Scorer:
    """Computes a bunch of metrics"""

    def __init__(
            self,
            n_labels: int
    ):
        self.n_labels = n_labels
        self.confusion_matrix = np.zeros((n_labels, n_labels))
        self.counter = 0
        self.eps = 1e-8
        self.losses = defaultdict(list)

    @property
    def acc(self):
        return self.confusion_matrix.trace() / (self.confusion_matrix.sum() + self.eps)

    @property
    def precision(self):
        return self.confusion_matrix.diagonal() / (self.confusion_matrix.sum(axis=-1) + self.eps)

    @property
    def macro_precision(self):
        return np.mean(self.precision)

    @property
    def micro_precision(self):
        return self.acc

    @property
    def weighted_precision(self):
        return (self.confusion_matrix.sum(axis=0) * self.precision) / (self.confusion_matrix.sum() + self.eps)

    @property
    def recall(self):
        return self.confusion_matrix.diagonal() / (self.confusion_matrix.sum(axis=0) + self.eps)

    @property
    def macro_recall(self):
        return np.mean(self.recall)

    @property
    def micro_recall(self):
        return self.acc

    @property
    def weighted_recall(self):
        return (self.confusion_matrix.sum(axis=0) * self.recall) / (self.confusion_matrix.sum() + self.eps)

    @property
    def f1(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall + self.eps)

    @property
    def macro_f1(self):
        return np.mean(self.f1)

    @property
    def micro_f1(self):
        return self.acc

    @property
    def weighted_f1(self):
        return (self.confusion_matrix.sum(axis=0) * self.f1) / (self.confusion_matrix.sum() + self.eps)

    def _inc_counter(self):
        self.counter += 1

    def add(self, logits: torch.Tensor, labels: torch.Tensor, loss: Union[LossType, float] = None,
            ignored_label: int = -100):
        self._inc_counter()
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(axis=-1)

        preds_np = preds.flatten().detach().cpu().numpy()
        labels_np = labels.flatten().detach().cpu().numpy()

        for y, y_hat in zip(labels_np, preds_np):
            if y != ignored_label:
                self.confusion_matrix[y][y_hat] += 1

        if loss is None:
            return
        elif isinstance(loss, float) or isinstance(loss, torch.Tensor):
            self.losses["global"].append(loss)
        elif isinstance(loss, KDLossOutput) or isinstance(loss, DistillationLoss):
            for attr in loss.__dataclass_fields__:
                self.losses[attr].append(getattr(loss, attr).detach())
        else:
            raise TypeError(f"Got 'loss' with type: {type(loss)}")

    def empty_losses(self):
        self.losses = defaultdict(list)

    def reset(self):
        """"""
        self.counter = 0
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.empty_losses()

    def to_dict(self) -> Dict[str, float]:
        return {"acc": self.acc, "p": self.precision, "micro-p": self.micro_precision,
                "macro-p": self.macro_precision, "weighted-p": self.weighted_precision,
                "r": self.recall, "micro-r": self.micro_recall, "macro-r": self.macro_recall,
                "weighted-r": self.weighted_recall, "f1": self.f1, "micro-f1": self.micro_f1,
                "macro-f1": self.macro_f1, "weighted-f1": self.weighted_f1}

    def get_table(self):
        table = []
        for k, v in self.to_dict().items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.float64):
                v = [v.item()]
            table.append([k] + v)
        return tabulate(table, headers=["metrics"] + [f"class {i}" for i in range(self.n_labels)])


class LooseScorer(Scorer):
    """
    Computes a bunch of 'loose' metrics.
    For some classification one might be interested in gathering some classes.

    E.g. an anomaly detection problem with different severity level:
    0: not an anomaly, 1: mild anomaly, 2: anomaly, 3: severe anomaly
    One could be interested in checking the performance of the loosen problem: anomaly/not anomaly.
    To do so:
    >>> loose_scorer = LooseScorer(loose_classes=[[0], [1,2,3]])
    """

    def __init__(self, loose_classes=List[List[Union[str, int]]]):
        super().__init__(n_labels=2)
        self.loose_classes = loose_classes

    @overrides
    def add(self, logits: torch.Tensor, labels: torch.Tensor, loss: Union[LossType, float] = None,
            ignored_label: int = -100):
        self._inc_counter()

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(axis=-1).flatten()
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
        elif isinstance(loss, KDLossOutput) or isinstance(loss, DistillationLoss):
            for attr in loss.__dataclass_fields__:
                self.losses[attr].append(getattr(loss, attr).detach())
        else:
            raise TypeError(f"Got 'loss' with type: {type(loss)}")
