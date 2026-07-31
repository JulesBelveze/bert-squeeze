from types import SimpleNamespace

import torch
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


class _LifecycleModule(BaseSequenceClassificationTransformerModule):
    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.model(input_ids.float())


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


def test_sequence_classification_lifecycle_uses_the_shared_step_contract(monkeypatch):
    monkeypatch.setattr(
        base_lt_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _DummyConfig(num_labels=kwargs["num_labels"]),
    )
    module = _LifecycleModule(
        training_config=OmegaConf.create(
            {
                "logging_steps": 2,
                "accumulation_steps": 1,
                "objective": "ce",
                "lr_scheduler": False,
            }
        ),
        pretrained_model="dummy",
        num_labels=3,
        model=nn.Linear(2, 3),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "labels": torch.tensor([0, 2]),
    }

    training_loss = module.training_step(batch, 0)
    training_loss.backward()
    validation_loss = module.validation_step(batch, 0)
    test_loss = module.test_step(batch, 0)
    probabilities = module.predict_step(batch, 0)

    assert module.model.weight.grad is not None
    assert torch.isfinite(training_loss)
    assert torch.isfinite(validation_loss)
    assert torch.isfinite(test_loss)
    assert module.validation_step_outputs[0]["logits"].shape == (2, 3)
    assert module.test_step_outputs[0]["labels"].tolist() == [0, 2]
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
