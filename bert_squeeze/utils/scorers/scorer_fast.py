from collections import defaultdict
from functools import partial
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from ..types import FastBertLoss


class FastBertScorer:
    """
    Helper class to compute various evaluation metrics for the FastBert models.

    This Scorer is different from the ones in the `scorer.py` as it computes metrics
    for the different branches of the model.
    We store metrics in a dict fashion where the key is the layer and the value are the different metrics.
    """

    def __init__(self, n_labels: int):
        """
        Args:
            n_labels (int):
                number of available labels
        """
        self.n_labels = n_labels
        self.confusion_matrix = defaultdict(partial(np.zeros, (n_labels, n_labels)))
        self.batch_counter = 0
        self.eps = 1e-8
        self.losses = defaultdict(list)

    @property
    def acc(self) -> Dict[str, float]:
        """
        Computes the accuracy score for the different layers:
            number of correct predictions / number of predictions

        Returns:
            Dict[str,float]: accuracy score for the different layers
        """
        accuracies = {}
        for layer, cfm in self.confusion_matrix.items():
            accuracies[layer] = cfm.trace() / (cfm.sum() + self.eps)
        return accuracies

    @property
    def precision(self) -> Dict[str, np.array]:
        """
        Computes the precision scores for every layer for each class using the following formula:
            tp / (tp + fp)

        Returns:
            Dict[str, np.array]: precision scores for the different layers
        """
        precisions = {}
        for layer, cfm in self.confusion_matrix.items():
            precisions[layer] = cfm.diagonal() / (cfm.sum(axis=-1) + self.eps)
        return precisions

    @property
    def recall(self) -> Dict[str, np.array]:
        """
        Computes the recall scores for every layer for each class using the following formula:
            tp / (tp + fn)

        Returns:
            Dict[str, np.array]: recall score for the different layers
        """
        recalls = {}
        for layer, cfm in self.confusion_matrix.items():
            recalls[layer] = cfm.diagonal() / (cfm.sum(axis=0) + self.eps)
        return recalls

    @property
    def f1(self) -> Dict[str, np.array]:
        """
        Computes the f1 scores for every layer for each class using the following formula:
            F1 = 2 * (precision * recall) / (precision + recall)

        Returns:
            Dict[str, np.array]: f1 scores for the different layers
        """
        accuracies = {}
        for layer, cfm in self.confusion_matrix.items():
            accuracies[layer] = (
                2
                * (self.precision[layer] * self.recall[layer])
                / (self.precision[layer] + self.recall[layer] + self.eps)
            )
        return accuracies

    def _inc_counter(self) -> None:
        """
        Increments the batch counter
        """
        self.batch_counter += 1

    def add(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        labels: torch.Tensor,
        loss: FastBertLoss = None,
        ignored_label: int = -100,
    ) -> None:
        """
        Updates the confusion matrix by using the predictions and the ground truth labels and keeps
        track of the loss.

        Args:
            logits (Union[torch.Tensor, List[torch.Tensor]]):
                predicted logits, depending on the training stage we are in they are either of type
                `torch.Tensor` (if `training_stage=0`) which corresponds to the logits of the last
                classifier. Or of type `List[torch.Tensor]` (if `training_stage=1`) which corresponds
                to the logits of all the layers.
            labels (torch.Tensor):
                ground truth labels
            loss (FastBertLoss):
                computed loss
            ignored_label (int):
                value to ignore during metric computation, default to -100 (PyTorch ignored value)
        """
        self._inc_counter()

        labels_np = labels.flatten().detach().cpu().numpy()
        if isinstance(logits, list):  # if we are in training_stage=1 (distillation stage)
            preds_np = []
            for elt in logits[:-1]:  # skipping the final classification layer
                pb = F.softmax(elt, dim=-1)
                pred = pb.argmax(dim=-1)
                preds_np.append(pred.flatten().detach().cpu().numpy())

            for layer_id, pred in enumerate(preds_np):
                for y, y_hat in zip(labels_np, pred):
                    if y != ignored_label:
                        self.confusion_matrix[f"branch_classifier_{layer_id}"][y][
                            y_hat
                        ] += 1
        else:
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            preds_np = preds.flatten().detach().cpu().numpy()

            for y, y_hat in zip(labels_np, preds_np):
                if y != ignored_label:
                    self.confusion_matrix["final_classifier"][y][y_hat] += 1

        if loss is None:
            return
        elif isinstance(loss, FastBertLoss):
            for attr in loss.__dataclass_fields__:
                if getattr(loss, attr) is None:
                    continue
                self.losses[attr].append(getattr(loss, attr).detach())
        else:
            raise TypeError(f"Got 'loss' with type: {type(loss)}")

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """"""
        return {
            "acc": {key: np.mean(val) for key, val in self.acc.items()},
            "prec": {key: np.mean(val) for key, val in self.precision.items()},
            "rec": {key: np.mean(val) for key, val in self.recall.items()},
            "f1": {key: np.mean(val) for key, val in self.f1.items()},
        }

    def reset(self) -> None:
        """
        Resets the confusion matrix, batch counter and losses.
        This method should be called after logging metrics.
        """
        self.batch_counter = 0
        self.confusion_matrix = defaultdict(
            partial(np.zeros, (self.n_labels, self.n_labels))
        )
        self.losses = defaultdict(list)

    def get_table(self) -> str:
        """
        Method to format all the metrics into a pretty table.

        Returns:
            str: prettyfied table summarizing all the metrics
        """
        table = []
        for metric, values in self.to_dict().items():
            for layer, val in values.items():
                table.append([f"{metric}_{layer}", val])
        return tabulate(
            table,
            headers=["metrics"] + [f"class {i}" for i in range(self.n_labels)],
            tablefmt="fancy_grid",
        )
