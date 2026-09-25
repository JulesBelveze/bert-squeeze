from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf

import bert_squeeze.models.base_lt_module as base_lt_module
from bert_squeeze.models.lt_automodel import LtSequenceClassificationAutoModel


class _SeqOutput:
    def __init__(self, logits: torch.Tensor, attentions=None):
        self.logits = logits
        self.attentions = attentions


class _TinyAutoModel(nn.Module):
    """Stand-in for ``AutoModelForSequenceClassification`` (no network)."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.embeddings = nn.Embedding(50, 8)
        self.classifier = nn.Linear(8, num_labels)
        # `AutoModelForSequenceClassification` exposes the backbone via `base_model`.
        self.base_model = self.embeddings
        self.config = SimpleNamespace(num_labels=num_labels)

    def forward(
        self, input_ids=None, attention_mask=None, output_attentions=False, **kwargs
    ):
        pooled = self.embeddings(input_ids).mean(dim=1)
        logits = self.classifier(pooled)
        return _SeqOutput(logits, attentions=() if output_attentions else None)


def _build_module(monkeypatch, num_labels: int) -> LtSequenceClassificationAutoModel:
    monkeypatch.setattr(
        base_lt_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(num_labels=kwargs["num_labels"]),
    )
    return LtSequenceClassificationAutoModel(
        training_config=OmegaConf.create(
            {
                "logging_steps": 2,
                "accumulation_steps": 1,
                "objective": "ce",
                "lr_scheduler": False,
            }
        ),
        pretrained_model="dummy",
        num_labels=num_labels,
        model=_TinyAutoModel(num_labels=num_labels),
    )


def test_forward_returns_logits(monkeypatch):
    module = _build_module(monkeypatch, num_labels=4)
    logits = module.forward(input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert logits.shape == (2, 4)


def test_encoder_is_backbone_without_head(monkeypatch):
    module = _build_module(monkeypatch, num_labels=3)
    assert module.encoder is module.model.base_model


def test_lifecycle_uses_shared_step_contract(monkeypatch):
    module = _build_module(monkeypatch, num_labels=3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([0, 2]),
    }

    training_loss = module.training_step(batch, 0)
    training_loss.backward()
    validation_loss = module.validation_step(batch, 0)
    probabilities = module.predict_step(batch, 0)

    assert torch.isfinite(training_loss)
    assert torch.isfinite(validation_loss)
    assert module.validation_step_outputs[0]["logits"].shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
