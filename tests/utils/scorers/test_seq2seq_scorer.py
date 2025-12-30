from unittest.mock import MagicMock

import pytest

from bert_squeeze.utils.scorers import Seq2SeqScorer


def _stub_loader(monkeypatch, mapping):
    def _loader(name):
        if name not in mapping:
            raise AssertionError(f"Unexpected metric load request: {name}")
        return mapping[name]

    monkeypatch.setattr(
        "bert_squeeze.utils.scorers.seq2seq_scorer.load",
        _loader,
    )


def test_compute_multiple_metrics(monkeypatch):
    rouge_metric = MagicMock()
    rouge_metric.compute.return_value = {"rouge1": 0.5, "rouge2": 0.25, "rougeL": 0.4}
    bleu_metric = MagicMock()
    bleu_metric.compute.return_value = {"score": 32.1}
    meteor_metric = MagicMock()
    meteor_metric.compute.return_value = {"meteor": 0.61}
    _stub_loader(
        monkeypatch,
        {"rouge": rouge_metric, "sacrebleu": bleu_metric, "meteor": meteor_metric},
    )

    scorer = Seq2SeqScorer(
        metrics=["rouge", "bleu", "meteor"],
        compute_length_stats=False,
    )
    scores = scorer.compute(["a"], ["b"])

    assert scores["rouge1"] == pytest.approx(0.5)
    assert scores["bleu"] == pytest.approx(32.1)
    assert scores["meteor"] == pytest.approx(0.61)


def test_group_metrics(monkeypatch):
    rouge_metric = MagicMock()
    rouge_metric.compute.return_value = {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.3}
    _stub_loader(monkeypatch, {"rouge": rouge_metric})

    predictions = ["a", "b", "c"]
    references = ["x", "y", "z"]
    group_labels = ["EN", "FR", "EN"]

    scorer = Seq2SeqScorer(metrics=["rouge"], compute_length_stats=False)
    scores = scorer.compute(predictions, references, group_labels=group_labels)

    assert "en_rouge1" in scores
    assert "fr_rouge1" in scores
    assert scores["en_rouge1"] == pytest.approx(0.4)


def test_length_stats_disabled(monkeypatch):
    rouge_metric = MagicMock()
    rouge_metric.compute.return_value = {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.3}
    _stub_loader(monkeypatch, {"rouge": rouge_metric})

    scorer = Seq2SeqScorer(metrics=["rouge"], compute_length_stats=False)
    scores = scorer.compute(["a"], ["b"])

    assert all(not key.startswith("length_") for key in scores)


def test_bertscore_kwargs_forwarded(monkeypatch):
    bertscore_metric = MagicMock()
    bertscore_metric.compute.return_value = {
        "precision": 0.7,
        "recall": 0.6,
        "f1": 0.65,
    }
    _stub_loader(monkeypatch, {"bertscore": bertscore_metric})

    scorer = Seq2SeqScorer(
        metrics=["bertscore"],
        compute_length_stats=False,
        bertscore_kwargs={"lang": "fr", "model_type": "xlm-roberta-base"},
    )
    scorer.compute(["foo"], ["bar"])

    bertscore_metric.compute.assert_called_once_with(
        predictions=["foo"],
        references=["bar"],
        lang="fr",
        model_type="xlm-roberta-base",
    )


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        Seq2SeqScorer(metrics=["unknown"])
