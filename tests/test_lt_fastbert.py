import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoConfig

import bert_squeeze.models.base_lt_module as base_lt_module
import bert_squeeze.models.lt_fastbert as lt_fastbert
from bert_squeeze.models.lt_fastbert import LtFastBert

TRAIN_CFG = OmegaConf.create(
    {
        "logging_steps": 2,
        "accumulation_steps": 1,
        "objective": "ce",
        "lr_scheduler": False,
    }
)


def _tiny_bert_config():
    return AutoConfig.for_model(
        "bert",
        num_labels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=100,
    )


def _build(monkeypatch, backbone_factory):
    """Construct LtFastBert offline: tiny config, dummy backbone, no file I/O."""
    monkeypatch.setattr(
        base_lt_module.AutoConfig, "from_pretrained", lambda *a, **k: _tiny_bert_config()
    )
    monkeypatch.setattr(
        lt_fastbert.AutoModel, "from_pretrained", lambda *a, **k: backbone_factory()
    )
    # A dummy `model` skips the AutoModelForSequenceClassification download in the base.
    return LtFastBert(
        training_config=TRAIN_CFG,
        num_labels=3,
        pretrained_model="dummy",
        model=nn.Module(),
    )


def test_attn_implementation_is_set_and_construction_succeeds(monkeypatch):
    """transformers >= 4.48 indexes attention classes by _attn_implementation; the guard
    must set it so FastBertGraph builds without KeyError(None)."""
    module = _build(monkeypatch, nn.Module)
    assert module.model_config._attn_implementation == "eager"


def test_load_pretrained_bert_model_uses_in_memory_state_dict(monkeypatch):
    """The default path reads weights from the model, not torch.load of a .bin file."""
    captured = {}

    class _Backbone(nn.Module):
        def state_dict(self, *args, **kwargs):
            captured["called"] = True
            return {}

    # Fail loudly if the old .bin round-trip (torch.load) is ever reintroduced.
    monkeypatch.setattr(
        lt_fastbert.torch,
        "load",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("torch.load must not run")),
    )

    _build(monkeypatch, _Backbone)

    assert captured.get("called") is True
