from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

from bert_squeeze.models.custom_transformers.berxit import BerxitModel
from bert_squeeze.models.custom_transformers.deebert import DeeBertModel
from bert_squeeze.models.custom_transformers.fastbert import FastBertGraph
from bert_squeeze.models.custom_transformers.theseus_bert import TheseusBertModel
from bert_squeeze.models.lt_berxit import LtBerxit
from bert_squeeze.models.lt_deebert import LtDeeBert
from bert_squeeze.models.lt_distilbert import LtCustomDistilBert
from bert_squeeze.models.lt_fastbert import LtFastBert
from bert_squeeze.models.lt_theseus_bert import LtTheseusBert
from bert_squeeze.utils.scorers import FastBertSequenceClassificationScorer


def _bert_config(num_hidden_layers: int = 1, num_labels: int = 2) -> BertConfig:
    return BertConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=num_hidden_layers,
        num_labels=num_labels,
        vocab_size=32,
    )


def _training_config(**overrides):
    config = {
        "logging_steps": 2,
        "accumulation_steps": 1,
        "objective": "ce",
        "lr_scheduler": False,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "token_type_ids": torch.zeros((2, 4), dtype=torch.long),
    }


def test_deebert_loads_and_uses_a_custom_pretrained_encoder(tmp_path):
    source_encoder = DeeBertModel(_bert_config(num_hidden_layers=2))
    source_encoder.save_pretrained(tmp_path)

    module = LtDeeBert(
        training_config=_training_config(train_highway=True, early_exit_entropy=-1.0),
        pretrained_model=str(tmp_path),
        num_labels=2,
    )
    module.eval()
    logits, _, _ = module(**_inputs())
    loss = module.test_step({**_inputs(), "labels": torch.tensor([0, 1])}, 0)
    predictions = module.predict_step(_inputs(), 0)

    assert module.model is module.bert
    assert module.bert.encoder.layer[0] is not module.bert.encoder.layer[1]
    assert module.bert.encoder.ramp[0] is not module.bert.encoder.ramp[1]
    assert torch.equal(
        module.bert.embeddings.word_embeddings.weight,
        source_encoder.embeddings.word_embeddings.weight,
    )
    assert logits.shape == (2, 2)
    assert predictions.shape == (2, 2)
    assert torch.isfinite(loss)


def test_berxit_loads_and_uses_a_custom_pretrained_encoder(tmp_path):
    source_encoder = BerxitModel(_bert_config(num_hidden_layers=2))
    source_encoder.save_pretrained(tmp_path)

    module = LtBerxit(
        training_config=_training_config(train_highway=True, early_exit_entropy=-1.0),
        pretrained_model=str(tmp_path),
        num_labels=2,
    )
    module.eval()
    logits, _, _, _ = module(**_inputs())
    loss = module.test_step({**_inputs(), "labels": torch.tensor([0, 1])}, 0)
    predictions = module.predict_step(_inputs(), 0)

    assert module.model is module.bert
    assert torch.equal(
        module.bert.embeddings.word_embeddings.weight,
        source_encoder.embeddings.word_embeddings.weight,
    )
    assert logits.shape == (2, 2)
    assert predictions.shape == (2, 2)
    assert torch.isfinite(loss)


def test_distilbert_uses_injected_sequence_classifier(tmp_path):
    config = DistilBertConfig(
        vocab_size=32,
        dim=16,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        num_labels=3,
    )
    injected_model = DistilBertForSequenceClassification(config)
    injected_model.save_pretrained(tmp_path)
    module = LtCustomDistilBert(
        training_config=_training_config(),
        pretrained_model=str(tmp_path),
        num_labels=3,
        model=injected_model,
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "token_type_ids": torch.zeros((2, 3), dtype=torch.long),
        "labels": torch.tensor([0, 2]),
    }

    logits = module(batch["input_ids"], batch["attention_mask"])
    attention_logits, attentions = module(
        batch["input_ids"],
        batch["attention_mask"],
        output_attentions=True,
    )
    loss = module.training_step(batch, 0)
    loss.backward()

    assert module.model is injected_model
    assert logits.shape == (2, 3)
    assert attention_logits.shape == (2, 3)
    assert len(attentions) == config.n_layers
    assert injected_model.classifier.weight.grad is not None


def test_fastbert_inference_uses_configured_label_count() -> None:
    graph = FastBertGraph(_bert_config(num_hidden_layers=2, num_labels=3))
    embeddings = torch.randn(2, 4, 16)

    probabilities, _ = graph(
        embeddings=embeddings,
        attention_mask=torch.zeros(2, 1, 1, 4),
        device="cpu",
        inference=True,
        inference_speed=1.1,
    )

    assert probabilities.shape == (2, 3)


def _fastbert_module(tmp_path: Path) -> LtFastBert:
    config = _bert_config(num_hidden_layers=2)
    injected_model = BertForSequenceClassification(config)
    injected_model.save_pretrained(tmp_path)

    return LtFastBert(
        training_config=_training_config(inference_speed=1.1),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=injected_model,
        scorer=FastBertSequenceClassificationScorer([0, 1]),
        training_stage=1,
    )


def test_fastbert_shared_lifecycle_handles_branch_logits(tmp_path: Path) -> None:
    module = _fastbert_module(tmp_path)
    batch = {**_inputs(), "labels": torch.tensor([0, 1])}

    training_loss = module.training_step(batch, 0)
    training_loss.backward()
    validation_loss = module.validation_step(batch, 0)
    test_loss = module.test_step(batch, 0)

    assert module.training_stage == 1
    assert (
        module.encoder.layer_classifiers["branch_classifier_0"].dense_logits.weight.grad
        is not None
    )
    assert torch.isfinite(training_loss)
    assert torch.isfinite(validation_loss)
    assert torch.isfinite(test_loss)
    assert "branch_classifier_0" in module.scorer.confusion_matrix
    assert module.validation_step_outputs[0]["logits"].shape == (2, 2)
    assert all(
        loss.device.type == "cpu" for loss in module.valid_scorer.losses["full_loss"]
    )


def test_fastbert_prediction_uses_adaptive_early_exit(tmp_path: Path) -> None:
    module = _fastbert_module(tmp_path)
    module.eval()
    with torch.no_grad():
        expected_probabilities, _ = module(
            **_inputs(), inference=True, inference_speed=1.1
        )

    layer_calls = [0] * len(module.encoder.layer)

    def count_layer(layer_idx: int) -> Callable[..., None]:
        def hook(*args: object) -> None:
            layer_calls[layer_idx] += 1

        return hook

    handles = [
        layer.register_forward_hook(count_layer(layer_idx))
        for layer_idx, layer in enumerate(module.encoder.layer)
    ]
    with torch.no_grad():
        probabilities = module.predict_step(_inputs(), 0)
    for handle in handles:
        handle.remove()

    assert torch.allclose(probabilities, expected_probabilities)
    assert layer_calls == [1, 0]


def test_theseus_loads_and_uses_a_custom_pretrained_encoder(tmp_path):
    source_encoder = TheseusBertModel(_bert_config(num_hidden_layers=6))
    source_encoder.save_pretrained(tmp_path)

    module = LtTheseusBert(
        training_config=_training_config(),
        pretrained_model=str(tmp_path),
        num_labels=2,
        replacement_scheduler=OmegaConf.create(
            {"type": "constant", "replacing_rate": 1.0}
        ),
    )
    logits = module(**_inputs())
    loss = module.training_step(
        {**_inputs(), "labels": torch.tensor([0, 1])},
        0,
    )

    assert module.model is module.encoder
    assert torch.equal(
        module.encoder.embeddings.word_embeddings.weight,
        source_encoder.embeddings.word_embeddings.weight,
    )
    assert logits.shape == (2, 2)
    assert torch.isfinite(loss)
    assert module.replacement_scheduler.step_counter == 1
