from unittest.mock import MagicMock

from bert_squeeze.utils.scorers import LightningSeq2SeqScorer


def _stub_metric(monkeypatch, return_value):
    metric = MagicMock()
    metric.compute.return_value = return_value

    def loader(name):
        return metric

    monkeypatch.setattr(
        "bert_squeeze.utils.scorers.seq2seq_scorer.load",
        loader,
    )
    return metric


def test_lightning_scorer_prefixes_keys(monkeypatch):
    _stub_metric(monkeypatch, {"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4})
    scorer = LightningSeq2SeqScorer(
        prefix="val",
        metrics=["rouge"],
        compute_length_stats=False,
    )

    scores = scorer.get_log_dict(
        predictions=["a", "b"],
        references=["x", "y"],
        group_labels=["en", "en"],
    )

    assert all(key.startswith("val_") for key in scores)


def test_lightning_scorer_handles_empty_prefix(monkeypatch):
    metric = _stub_metric(monkeypatch, {"rouge1": 0.5})
    scorer = LightningSeq2SeqScorer(
        prefix="",
        metrics=["rouge"],
        compute_length_stats=False,
    )

    def fake_compute(*_, **__):
        return {"rouge1": 0.75}

    metric.compute.side_effect = fake_compute
    scores = scorer.get_log_dict(predictions=["a"], references=["x"])

    assert scores == {"rouge1": 0.75}
