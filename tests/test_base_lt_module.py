from types import SimpleNamespace

import torch.nn as nn
from omegaconf import OmegaConf

import bert_squeeze.models.base_lt_module as base_lt_module
from bert_squeeze.models.base_lt_module import BaseSequenceClassificationTransformerModule


class _DummyConfig:
    def __init__(self, num_labels: int):
        self.num_labels = num_labels


class _DummyModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.config = SimpleNamespace(num_labels=num_labels)


def test_base_sequence_classification_transformer_module_default_scorer(monkeypatch):
    def _fake_from_pretrained(*args, **kwargs):
        return _DummyConfig(num_labels=kwargs["num_labels"])

    monkeypatch.setattr(
        base_lt_module.AutoConfig, "from_pretrained", _fake_from_pretrained
    )

    training_config = OmegaConf.create(
        {
            "logging_steps": 2,
            "accumulation_steps": 1,
            "objective": "ce",
            "lr_scheduler": False,
        }
    )

    module = BaseSequenceClassificationTransformerModule(
        training_config=training_config,
        pretrained_model="dummy",
        num_labels=3,
        model=_DummyModel(num_labels=3),
        scorer=None,
    )

    assert module.scorer.labels == [0, 1, 2]
    assert module.scorer.n_labels == 3
    assert module.scorer is not module.valid_scorer
    assert module.scorer is not module.test_scorer
