from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import pytest
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification

from bert_squeeze.assistants.train_assistant import TrainAssistant
from bert_squeeze.models.custom_transformers.layer_dropout import LayerDropoutWrapper
from bert_squeeze.models.lt_layerskip import LtLayerSkip
from bert_squeeze.utils.callbacks.layerskip_curriculum import LayerSkipCurriculumCallback


class _IncrementLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.increment = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.attention_masks: list[Optional[torch.Tensor]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor]:
        self.calls += 1
        self.batch_sizes.append(hidden_states.shape[0])
        self.attention_masks.append(attention_mask)
        return (hidden_states * self.scale + self.increment,)


def _training_config() -> DictConfig:
    return OmegaConf.create(
        {
            "logging_steps": 10,
            "accumulation_steps": 1,
            "discriminative_learning": False,
            "weight_decay": 0.0,
            "learning_rates": [0.1],
            "adam_eps": 1e-8,
            "lr_scheduler": False,
            "optimizer": "sgd",
        }
    )


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> str:
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=4,
        num_labels=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    BertForSequenceClassification(config).save_pretrained(tmp_path)
    return str(tmp_path)


def _inputs() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).long(),
        "token_type_ids": torch.zeros_like(input_ids),
    }


def _build_model(
    checkpoint: str,
    *,
    p_max: float = 0.2,
    e_scale: float = 0.2,
    exit_layer: int = 2,
    inference_mode: bool = False,
    dropout_schedule: str = "exponential",
) -> LtLayerSkip:
    return LtLayerSkip(
        training_config=_training_config(),
        pretrained_model=checkpoint,
        num_labels=2,
        p_max=p_max,
        e_scale=e_scale,
        exit_layer=exit_layer,
        inference_mode=inference_mode,
        dropout_schedule=dropout_schedule,
    )


def test_layer_dropout_skips_computation_and_restores_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.zeros(2, 3, 4, requires_grad=True)
    attention_mask = torch.arange(6).reshape(2, 1, 1, 3)

    dropped_layer = _IncrementLayer()
    fully_dropped = LayerDropoutWrapper(dropped_layer, dropout_prob=1.0)
    fully_dropped.train()
    output = fully_dropped(hidden_states, attention_mask)

    assert dropped_layer.calls == 0
    assert torch.equal(output[0], hidden_states)
    output[0].sum().backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in dropped_layer.parameters()
    )
    hidden_states.grad = None

    fully_dropped.eval()
    output = fully_dropped(hidden_states, attention_mask)

    assert dropped_layer.calls == 1
    assert torch.equal(output[0], hidden_states + 1)

    monkeypatch.setattr(
        torch,
        "rand",
        lambda batch_size, device: torch.tensor([0.1, 0.9], device=device),
    )
    partial_layer = _IncrementLayer()
    partially_dropped = LayerDropoutWrapper(partial_layer, dropout_prob=0.5)
    partially_dropped.train()
    output = partially_dropped(hidden_states, attention_mask)

    assert partial_layer.calls == 1
    assert partial_layer.batch_sizes == [1]
    assert torch.equal(partial_layer.attention_masks[0], attention_mask[1:])
    assert torch.equal(output[0][0], hidden_states[0])
    assert torch.equal(output[0][1], hidden_states[1] + 1)
    output[0].sum().backward()
    assert partial_layer.increment.grad is not None
    assert torch.equal(hidden_states.grad, torch.ones_like(hidden_states))


def test_layer_and_loss_scales_follow_the_paper(tiny_checkpoint: str) -> None:
    model = _build_model(tiny_checkpoint)

    expected_dropout = [0.2 * (2 ** (layer_idx / 3) - 1) for layer_idx in range(4)]
    expected_loss = torch.tensor([0.0, 0.2, 0.6, 3.6])
    expected_loss /= expected_loss.sum()

    assert model._compute_dropout_schedule() == pytest.approx(expected_dropout)
    assert torch.allclose(model.loss_scales, expected_loss)


def test_early_exit_returns_logits_without_running_later_layers(
    tiny_checkpoint: str,
) -> None:
    model = _build_model(tiny_checkpoint, inference_mode=True, exit_layer=2)
    model.eval()
    call_counts = [0] * model.num_layers
    inputs = _inputs()
    hidden_states = model._forward_all_layers(**inputs)
    expected_logits = model._get_layer_logits(hidden_states[model.exit_layer - 1])

    def count_layer(layer_idx: int) -> Callable[..., None]:
        def hook(*args: object) -> None:
            call_counts[layer_idx] += 1

        return hook

    for layer_idx, layer in enumerate(model._get_transformer_layers()):
        layer.register_forward_hook(count_layer(layer_idx))

    logits = model(**inputs)

    assert logits.shape == (2, 2)
    assert torch.allclose(logits, expected_logits, atol=1e-6)
    assert call_counts == [1, 1, 0, 0]


def test_early_exit_accepts_input_embeddings(tiny_checkpoint: str) -> None:
    model = _build_model(tiny_checkpoint, inference_mode=True, exit_layer=2)
    model.eval()
    inputs = _inputs()
    input_embeddings = model._get_base_model().embeddings.word_embeddings(
        inputs["input_ids"]
    )
    embedded_inputs = {
        "inputs_embeds": input_embeddings,
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
    }
    hidden_states = model._forward_all_layers(
        input_ids=None,
        **embedded_inputs,
    )
    expected_logits = model._get_layer_logits(hidden_states[model.exit_layer - 1])

    logits = model(**embedded_inputs)

    assert torch.allclose(logits, expected_logits, atol=1e-6)
    with pytest.raises(ValueError, match="either input_ids or inputs_embeds"):
        model(input_ids=inputs["input_ids"], **embedded_inputs)


def test_standard_forward_matches_bert_classifier(tiny_checkpoint: str) -> None:
    model = _build_model(tiny_checkpoint)
    model.eval()
    inputs = _inputs()

    logits = model(**inputs)
    reference_logits = model.model(**inputs).logits

    assert logits.shape == (2, 2)
    assert torch.allclose(logits, reference_logits, atol=1e-6)


def test_training_loss_only_uses_curriculum_exits(tiny_checkpoint: str) -> None:
    model = _build_model(tiny_checkpoint, p_max=0.0)
    model.eval()
    model.curriculum_mask.copy_(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    labels = torch.tensor([0, 1])
    outputs = model._forward_all_layers(**_inputs())
    classifier_calls = 0

    def count_classifier_calls(*args: object) -> None:
        nonlocal classifier_calls
        classifier_calls += 1

    handle = model.model.classifier.register_forward_hook(count_classifier_calls)
    actual_loss = model._compute_training_loss(outputs, labels)
    handle.remove()

    enabled_layers = [1, 3]
    expected_losses = torch.stack(
        [
            model.objective(model._get_layer_logits(outputs[layer_idx]), labels)
            for layer_idx in enabled_layers
        ]
    )
    weights = model.loss_scales[enabled_layers]
    expected_loss = (expected_losses * weights).sum() / weights.sum()

    assert classifier_calls == 2
    assert torch.allclose(actual_loss, expected_loss)


def test_curriculum_rotates_early_exits_and_keeps_final_layer() -> None:
    callback = LayerSkipCurriculumCallback(
        curriculum_type="rotational", rotation_period=3
    )
    module = SimpleNamespace(num_layers=4, curriculum_mask=torch.zeros(4))
    trainer = SimpleNamespace(
        global_step=1, max_steps=100, estimated_stepping_batches=100
    )

    callback.on_train_batch_start(trainer, module, None, 0)

    assert torch.equal(module.curriculum_mask, torch.tensor([0.0, 1.0, 0.0, 1.0]))

    gradual = LayerSkipCurriculumCallback(curriculum_type="gradual")
    trainer.global_step = 25
    gradual.on_train_batch_start(trainer, module, None, 0)

    assert torch.equal(module.curriculum_mask, torch.tensor([0.0, 0.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda path: _build_model(path, p_max=-0.1), id="p-max-low"),
        pytest.param(lambda path: _build_model(path, p_max=1.1), id="p-max-high"),
        pytest.param(lambda path: _build_model(path, e_scale=-0.1), id="e-scale"),
        pytest.param(lambda path: _build_model(path, exit_layer=0), id="exit-low"),
        pytest.param(lambda path: _build_model(path, exit_layer=5), id="exit-high"),
        pytest.param(
            lambda path: _build_model(path, dropout_schedule="unknown"),
            id="schedule",
        ),
    ],
)
def test_layerskip_rejects_invalid_configuration(
    tiny_checkpoint: str,
    factory: Callable[[str], LtLayerSkip],
) -> None:
    with pytest.raises(ValueError):
        factory(tiny_checkpoint)


def test_curriculum_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        LayerSkipCurriculumCallback(curriculum_type="unknown")
    with pytest.raises(ValueError):
        LayerSkipCurriculumCallback(rotation_period=0)


def test_layerskip_trains_with_synthetic_batches(tiny_checkpoint: str) -> None:
    model = _build_model(tiny_checkpoint, p_max=0.5)
    initial_classifier = model.model.classifier.weight.detach().clone()
    initial_encoder = (
        model._get_transformer_layers()[0]
        .layer.attention.self.query.weight.detach()
        .clone()
    )
    batch = {**_inputs(), "labels": torch.tensor([0, 1])}
    curriculum = LayerSkipCurriculumCallback(
        curriculum_type="rotational", rotation_period=3
    )
    trainer = Trainer(
        accelerator="cpu",
        callbacks=[curriculum],
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=2,
    )

    trainer.fit(model, train_dataloaders=DataLoader([batch, batch], batch_size=None))

    trained_encoder = model._get_transformer_layers()[0].layer.attention.self.query.weight
    assert trainer.global_step == 2
    assert torch.equal(model.curriculum_mask, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert not torch.equal(model.model.classifier.weight, initial_classifier)
    assert not torch.equal(trained_encoder, initial_encoder)


def test_layerskip_assistant_uses_the_local_checkpoint(tiny_checkpoint: str) -> None:
    assistant = TrainAssistant(
        "layerskip", model_kwargs={"pretrained_model": tiny_checkpoint}
    )

    assert isinstance(assistant.model, LtLayerSkip)
    assert isinstance(assistant.callbacks[0], LayerSkipCurriculumCallback)


def test_wrapped_layers_preserve_attention_and_pruning(tiny_checkpoint: str) -> None:
    model = _build_model(tiny_checkpoint)
    first_layer = model._get_transformer_layers()[0]

    model.model.prune_heads({0: [0]})

    assert first_layer.attention.self.num_attention_heads == 1


def test_fully_dropped_layers_preserve_attention_outputs(
    tiny_checkpoint: str,
) -> None:
    model = _build_model(tiny_checkpoint)
    model.train()
    for layer in model._get_transformer_layers():
        layer.dropout_prob = 1.0

    outputs = model._get_base_model()(
        **_inputs(), output_attentions=True, return_dict=True
    )

    assert outputs.attentions is not None
    assert len(outputs.attentions) == model.num_layers
    assert all(torch.count_nonzero(attention) == 0 for attention in outputs.attentions)


def test_layerskip_state_round_trips_through_hugging_face(
    tiny_checkpoint: str, tmp_path: Path
) -> None:
    model = _build_model(tiny_checkpoint)
    model.eval()
    expected_logits = model(**_inputs())
    export_path = tmp_path / "exported"

    model.model.save_pretrained(export_path)
    reloaded = BertForSequenceClassification.from_pretrained(export_path)
    reloaded.eval()

    assert torch.allclose(reloaded(**_inputs()).logits, expected_logits, atol=1e-6)

    restored = _build_model(tiny_checkpoint)
    restored.load_state_dict(model.state_dict())
    restored.eval()

    assert torch.allclose(restored(**_inputs()), expected_logits, atol=1e-6)
