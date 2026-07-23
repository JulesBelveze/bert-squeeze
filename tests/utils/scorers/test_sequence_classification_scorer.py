import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from bert_squeeze.utils.scorers import (
    BaseSequenceClassificationScorer,
    FastBertSequenceClassificationScorer,
)


def _classification_batch():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    predictions = torch.tensor([0, 1, 1, 1, 2, 0])
    logits = torch.nn.functional.one_hot(predictions, num_classes=3).float()
    return labels, predictions, logits


def test_sequence_classification_metrics_match_scikit_learn():
    labels, predictions, logits = _classification_batch()
    scorer = BaseSequenceClassificationScorer([0, 1, 2])
    scorer.add(logits, labels)

    labels_np = labels.numpy()
    predictions_np = predictions.numpy()
    np.testing.assert_allclose(
        scorer.precision,
        precision_score(labels_np, predictions_np, average=None, zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.recall,
        recall_score(labels_np, predictions_np, average=None, zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.f1,
        f1_score(labels_np, predictions_np, average=None, zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.weighted_precision,
        precision_score(labels_np, predictions_np, average="weighted", zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.weighted_recall,
        recall_score(labels_np, predictions_np, average="weighted", zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.weighted_f1,
        f1_score(labels_np, predictions_np, average="weighted", zero_division=0),
    )
    assert "weighted-p" in scorer.get_table()


def test_fastbert_metrics_match_scikit_learn():
    labels, predictions, logits = _classification_batch()
    scorer = FastBertSequenceClassificationScorer([0, 1, 2])
    scorer.add([logits, logits], labels)

    labels_np = labels.numpy()
    predictions_np = predictions.numpy()
    layer = "branch_classifier_0"
    np.testing.assert_allclose(
        scorer.precision[layer],
        precision_score(labels_np, predictions_np, average=None, zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.recall[layer],
        recall_score(labels_np, predictions_np, average=None, zero_division=0),
    )
    np.testing.assert_allclose(
        scorer.f1[layer],
        f1_score(labels_np, predictions_np, average=None, zero_division=0),
    )
