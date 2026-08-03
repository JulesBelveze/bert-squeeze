from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)

from bert_squeeze.models.custom_transformers.berxit import BerxitModel
from bert_squeeze.models.custom_transformers.deebert import DeeBertModel
from bert_squeeze.models.lt_berxit import LtBerxit
from bert_squeeze.models.lt_deebert import LtDeeBert
from bert_squeeze.utils.optimizers import (
    OptimizerParameterGroup,
    build_optimizer_parameter_groups,
)
from bert_squeeze.utils.types import RampOutput


class _Encoder(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(layer_count)])


class _LayeredModel(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(4, 2)
        self.encoder = _Encoder(layer_count)
        self.classifier = nn.Linear(2, 2)


def _learning_rate_for(
    groups: list[OptimizerParameterGroup], parameter: torch.Tensor
) -> Optional[float]:
    for group in groups:
        if any(group_parameter is parameter for group_parameter in group["params"]):
            return group.get("lr")
    raise AssertionError("Parameter is missing from optimizer groups.")


def _grouped_parameter_ids(
    groups: list[OptimizerParameterGroup],
) -> list[int]:
    return [id(parameter) for group in groups for parameter in group["params"]]


@pytest.mark.parametrize(
    ("learning_rates", "expected_rates", "embedding_rate"),
    [
        ([0.1], [0.025, 0.05, 0.1], 0.0125),
        ([0.01, 0.02, 0.03], [0.01, 0.02, 0.03], 0.005),
    ],
)
def test_optimizer_groups_follow_model_depth(
    learning_rates: list[float],
    expected_rates: list[float],
    embedding_rate: float,
) -> None:
    model = _LayeredModel(layer_count=3)

    groups = build_optimizer_parameter_groups(
        model.named_parameters(),
        discriminative_learning=True,
        learning_rates=learning_rates,
        layer_lr_decay=0.5,
        weight_decay=0.01,
    )

    assert [
        _learning_rate_for(groups, layer.weight) for layer in model.encoder.layer
    ] == pytest.approx(expected_rates)
    assert _learning_rate_for(groups, model.embeddings.weight) == pytest.approx(
        embedding_rate
    )
    assert _learning_rate_for(groups, model.classifier.weight) == pytest.approx(
        learning_rates[-1]
    )
    grouped_parameter_ids = _grouped_parameter_ids(groups)
    assert len(grouped_parameter_ids) == len(set(grouped_parameter_ids))
    assert len(grouped_parameter_ids) == len(list(model.parameters()))
    assert all(group["params"] for group in groups)


def test_optimizer_groups_reject_mismatched_layer_rates() -> None:
    model = _LayeredModel(layer_count=3)

    with pytest.raises(ValueError, match="Expected 3 layer learning rates"):
        build_optimizer_parameter_groups(
            model.named_parameters(),
            discriminative_learning=True,
            learning_rates=[0.1, 0.2],
            layer_lr_decay=0.5,
            weight_decay=0.01,
        )


def _t5_model(encoder_blocks: int, decoder_blocks: int) -> T5ForConditionalGeneration:
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=encoder_blocks,
            num_decoder_layers=decoder_blocks,
            num_heads=2,
        )
    )


@pytest.mark.parametrize(
    ("learning_rates", "expected_encoder_rates", "expected_decoder_rates"),
    [
        ([0.1], [0.05, 0.1], [0.0125, 0.025, 0.05, 0.1]),
        ([0.01, 0.02, 0.03, 0.04], [0.03, 0.04], [0.01, 0.02, 0.03, 0.04]),
    ],
)
def test_encoder_decoder_stacks_receive_independent_schedules(
    learning_rates: list[float],
    expected_encoder_rates: list[float],
    expected_decoder_rates: list[float],
) -> None:
    model = _t5_model(encoder_blocks=2, decoder_blocks=4)

    groups = build_optimizer_parameter_groups(
        model.named_parameters(),
        discriminative_learning=True,
        learning_rates=learning_rates,
        layer_lr_decay=0.5,
        weight_decay=0.01,
    )

    encoder_rates = [
        _learning_rate_for(groups, block.layer[0].SelfAttention.q.weight)
        for block in model.encoder.block
    ]
    decoder_rates = [
        _learning_rate_for(groups, block.layer[0].SelfAttention.q.weight)
        for block in model.decoder.block
    ]
    assert encoder_rates == pytest.approx(expected_encoder_rates)
    assert decoder_rates == pytest.approx(expected_decoder_rates)


@pytest.mark.parametrize(
    "architecture",
    ["gpt2", "llama"],
)
def test_normalization_parameters_do_not_use_weight_decay(
    architecture: str,
) -> None:
    if architecture == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                n_layer=2,
                n_head=2,
                n_embd=16,
                n_positions=16,
                vocab_size=32,
            )
        )
    else:
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=32,
                max_position_embeddings=16,
            )
        )

    groups = build_optimizer_parameter_groups(
        model.named_parameters(),
        discriminative_learning=True,
        learning_rates=[0.1],
        layer_lr_decay=0.5,
        weight_decay=0.01,
    )
    weight_decay_by_parameter = {
        id(parameter): group["weight_decay"]
        for group in groups
        for parameter in group["params"]
    }
    normalization_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "ln_" in name or "layernorm" in name or name.endswith("norm.weight")
    ]

    assert normalization_parameters
    assert all(
        weight_decay_by_parameter[id(parameter)] == 0.0
        for parameter in normalization_parameters
    )


def _model_config(tmp_path: Path) -> BertConfig:
    model_config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=2,
    )
    model_config.save_pretrained(tmp_path)
    return model_config


def _training_config(**overrides: object) -> DictConfig:
    config = {
        "logging_steps": 2,
        "accumulation_steps": 1,
        "objective": "ce",
        "lr_scheduler": False,
        "optimizer": "adamw",
        "adam_eps": 1e-8,
        "discriminative_learning": True,
        "learning_rates": [0.1],
        "layer_lr_decay": 0.5,
        "weight_decay": 0.01,
        "train_highway": True,
        "train_gates": True,
        "train_stage": "backbone",
        "early_exit_entropy": -1.0,
        "gate_thresholds": 0.5,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3], [3, 2, 1]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }


@pytest.mark.parametrize("train_highway", [False, True])
def test_deebert_training_stages_update_the_inference_exits(
    tmp_path: Path, train_highway: bool
) -> None:
    model_config = _model_config(tmp_path)
    module = LtDeeBert(
        training_config=_training_config(train_highway=train_highway),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=DeeBertModel(model_config),
    )

    output = module._classification_output(_batch())
    module._classification_loss(output, _batch()["labels"]).backward()
    grouped_ids = set(_grouped_parameter_ids(module._get_optimizer_parameters()))
    trainable_ids = {
        id(parameter) for parameter in module.parameters() if parameter.requires_grad
    }

    backbone_grad = module.bert.encoder.layer[0].attention.self.query.weight.grad
    intermediate_grad = module.bert.encoder.ramp[0].classifier.weight.grad
    final_grad = module.bert.encoder.ramp[-1].classifier.weight.grad
    if train_highway:
        assert backbone_grad is None
        assert intermediate_grad is not None
        assert final_grad is None
    else:
        assert backbone_grad is not None
        assert intermediate_grad is None
        assert final_grad is not None
    assert grouped_ids == trainable_ids


def _berxit_module(tmp_path: Path, **overrides: object) -> LtBerxit:
    model_config = _model_config(tmp_path)
    return LtBerxit(
        training_config=_training_config(**overrides),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=BerxitModel(model_config),
    )


def test_berxit_gate_stage_only_trains_the_shared_gate(tmp_path: Path) -> None:
    module = _berxit_module(tmp_path, train_stage="gates")

    trainable_parameters = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    grouped_ids = set(_grouped_parameter_ids(module._get_optimizer_parameters()))

    assert trainable_parameters
    assert all("gates" in name for name in trainable_parameters)
    assert grouped_ids == {
        id(parameter) for parameter in module.parameters() if parameter.requires_grad
    }


def test_berxit_trains_final_and_intermediate_exits(tmp_path: Path) -> None:
    module = _berxit_module(tmp_path)
    batch = _batch()
    output = module._classification_output(batch)

    module.loss(
        labels=batch["labels"],
        ramps_exits=output.ramps_exits,
        train_ramps=False,
    ).backward()
    assert module.bert.encoder.ramp[0].classifier.weight.grad is None
    assert module.bert.encoder.ramp[-1].classifier.weight.grad is not None

    module.zero_grad(set_to_none=True)
    output = module._classification_output(batch)
    module.loss(
        labels=batch["labels"],
        ramps_exits=output.ramps_exits,
        train_ramps=True,
    ).backward()
    assert module.bert.encoder.ramp[0].classifier.weight.grad is not None
    assert module.bert.encoder.ramp[-1].classifier.weight.grad is not None


def test_berxit_uses_label_based_certainty_targets(tmp_path: Path) -> None:
    module = _berxit_module(tmp_path)
    model_config = module.model_config
    labels = torch.tensor([0])
    pooled_output = torch.zeros(1, model_config.hidden_size)
    correct_early_exit = RampOutput(
        logits=torch.tensor([[4.0, -4.0]]), pooled_output=pooled_output
    )
    wrong_final_exit = RampOutput(
        logits=torch.tensor([[-4.0, 4.0]]), pooled_output=pooled_output
    )
    early_gate = torch.tensor([[2.0]], requires_grad=True)
    final_gate = torch.tensor([[2.0]], requires_grad=True)

    loss = module.loss(
        labels=labels,
        ramps_exits=[correct_early_exit, wrong_final_exit],
        gates_logits=(early_gate, final_gate),
        train_ramps=True,
        train_gates=True,
    )
    loss.backward()

    assert early_gate.grad is not None and early_gate.grad.item() < 0
    assert final_gate.grad is not None and final_gate.grad.item() > 0
    assert isinstance(module.bert.encoder.gates, nn.Linear)


def test_berxit_inference_returns_every_sample_after_early_exit(tmp_path: Path) -> None:
    module = _berxit_module(tmp_path, gate_thresholds=0.0)
    module.bert.set_inference_mode(inference=True)
    batch = _batch()

    logits, ramps_exits, exit_layer, _ = module.forward(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    assert logits.shape == (2, 2)
    assert len(ramps_exits) == 2
    assert exit_layer == 0


def test_plateau_scheduler_uses_epoch_training_loss(tmp_path: Path) -> None:
    model_config = _model_config(tmp_path)
    module = LtDeeBert(
        training_config=_training_config(
            train_highway=False,
            lr_scheduler=True,
        ),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=DeeBertModel(model_config),
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_epochs=1,
    )

    trainer.fit(module, train_dataloaders=DataLoader([_batch()], batch_size=None))

    assert "train/epoch_loss" in trainer.callback_metrics
