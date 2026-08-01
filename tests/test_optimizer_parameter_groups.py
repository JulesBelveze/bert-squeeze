from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertConfig, T5Config, T5ForConditionalGeneration

from bert_squeeze.models.custom_transformers.berxit import BerxitModel
from bert_squeeze.models.custom_transformers.deebert import DeeBertModel
from bert_squeeze.models.lt_berxit import LtBerxit
from bert_squeeze.models.lt_deebert import LtDeeBert
from bert_squeeze.utils.optimizers import (
    OptimizerParameterGroup,
    build_optimizer_parameter_groups,
    register_legacy_optimizer_state_migration,
)
from bert_squeeze.utils.schedulers import GroupCompatibleReduceLROnPlateau

_NO_DECAY_NAMES = ("bias", "gamma", "beta", "LayerNorm.weight", "layer_norm.weight")


class _Encoder(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(layer_count)])


class _LayeredModel(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.encoder = _Encoder(layer_count)
        self.classifier = nn.Linear(2, 2)


def _learning_rate_for(
    groups: list[OptimizerParameterGroup], parameter: nn.Parameter
) -> Optional[float]:
    for group in groups:
        if any(group_parameter is parameter for group_parameter in group["params"]):
            return group.get("lr")
    raise AssertionError("Parameter is missing from optimizer groups.")


@pytest.mark.parametrize(
    ("learning_rates", "expected_rates"),
    [
        ([0.1], [0.025, 0.05, 0.1]),
        ([0.01, 0.02, 0.03], [0.01, 0.02, 0.03]),
    ],
)
def test_optimizer_groups_follow_model_depth(
    learning_rates: list[float], expected_rates: list[float]
) -> None:
    model = _LayeredModel(layer_count=3)

    groups = build_optimizer_parameter_groups(
        model.named_parameters(),
        discriminative_learning=True,
        learning_rates=learning_rates,
        layer_lr_decay=0.5,
        weight_decay=0.01,
    )

    actual_rates = [
        _learning_rate_for(groups, layer.weight) for layer in model.encoder.layer
    ]
    assert actual_rates == pytest.approx(expected_rates)
    assert [groups[index]["lr"] for index in range(3)] == pytest.approx(expected_rates)
    assert [groups[12 + index]["lr"] for index in range(3)] == pytest.approx(
        expected_rates
    )
    assert all("lr" not in group for group in groups[24:])
    assert _learning_rate_for(groups, model.classifier.weight) is None
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


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


def _legacy_optimizer_groups(model: nn.Module) -> list[OptimizerParameterGroup]:
    named_parameters = list(model.named_parameters())
    layer_keys = [f"layer.{index}." for index in range(12)]
    legacy_rates = [0.1 * pow(0.5, 11 - index) for index in range(12)]
    legacy_groups: list[OptimizerParameterGroup] = []
    for use_weight_decay in (True, False):
        legacy_groups.extend(
            OptimizerParameterGroup(
                params=[
                    parameter
                    for name, parameter in named_parameters
                    if layer_key in name
                    and (not any(no_decay in name for no_decay in _NO_DECAY_NAMES))
                    == use_weight_decay
                ],
                weight_decay=0.01 if use_weight_decay else 0.0,
                lr=legacy_rates[index],
            )
            for index, layer_key in enumerate(layer_keys)
        )
    for use_weight_decay in (True, False):
        legacy_groups.append(
            OptimizerParameterGroup(
                params=[
                    parameter
                    for name, parameter in named_parameters
                    if not any(layer_key in name for layer_key in layer_keys)
                    and (not any(no_decay in name for no_decay in _NO_DECAY_NAMES))
                    == use_weight_decay
                ],
                weight_decay=0.01 if use_weight_decay else 0.0,
            )
        )
    return legacy_groups


def _current_optimizer(model: nn.Module) -> AdamW:
    optimizer = AdamW(
        build_optimizer_parameter_groups(
            model.named_parameters(),
            discriminative_learning=True,
            learning_rates=[0.1],
            layer_lr_decay=0.5,
            weight_decay=0.01,
        ),
        lr=0.1,
    )
    register_legacy_optimizer_state_migration(optimizer, model.named_parameters())
    return optimizer


@pytest.mark.parametrize("layer_count", [3, 24])
def test_optimizer_groups_restore_legacy_optimizer_state(layer_count: int) -> None:
    model = _LayeredModel(layer_count=layer_count)
    legacy_optimizer = AdamW(
        _legacy_optimizer_groups(model),
        lr=0.1,
        betas=(0.8, 0.88),
        eps=1e-6,
    )
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    legacy_optimizer.step()
    legacy_optimizer.zero_grad()

    current_optimizer = _current_optimizer(model)
    current_optimizer.load_state_dict(legacy_optimizer.state_dict())
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    current_optimizer.step()

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert current_optimizer.param_groups[0]["betas"] == (0.8, 0.88)
    assert current_optimizer.param_groups[0]["eps"] == 1e-6


def test_scheduler_restores_after_optimizer_group_migration() -> None:
    model = _LayeredModel(layer_count=24)
    legacy_optimizer = AdamW(_legacy_optimizer_groups(model), lr=0.1)
    legacy_scheduler = ReduceLROnPlateau(legacy_optimizer, factor=0.5, patience=0)
    current_optimizer = _current_optimizer(model)
    current_optimizer.load_state_dict(legacy_optimizer.state_dict())
    current_scheduler = GroupCompatibleReduceLROnPlateau(
        current_optimizer, factor=0.5, patience=0
    )

    current_scheduler.load_state_dict(legacy_scheduler.state_dict())
    current_scheduler.step(1.0)
    current_scheduler.step(2.0)

    assert len(current_scheduler.min_lrs) == len(current_optimizer.param_groups)


def _t5_model(block_count: int) -> T5ForConditionalGeneration:
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=block_count,
            num_decoder_layers=block_count,
            num_heads=2,
        )
    )


def test_t5_optimizer_groups_follow_block_depth() -> None:
    model = _t5_model(block_count=2)

    groups = build_optimizer_parameter_groups(
        model.named_parameters(),
        discriminative_learning=True,
        learning_rates=[0.1],
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
    grouped_parameters = [parameter for group in groups for parameter in group["params"]]

    assert encoder_rates == pytest.approx([0.05, 0.1])
    assert decoder_rates == pytest.approx([0.05, 0.1])
    assert len(grouped_parameters) == len(
        {id(parameter) for parameter in grouped_parameters}
    )
    assert len(grouped_parameters) == len(list(model.parameters()))


def test_t5_legacy_state_migrates_when_group_counts_match() -> None:
    model = _t5_model(block_count=12)
    legacy_optimizer = AdamW(_legacy_optimizer_groups(model), lr=0.1)
    current_optimizer = _current_optimizer(model)

    assert len(legacy_optimizer.param_groups) == len(current_optimizer.param_groups)
    current_optimizer.load_state_dict(legacy_optimizer.state_dict())

    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    current_optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


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


def _ramp_training_config(**overrides: object) -> DictConfig:
    config = {
        "logging_steps": 2,
        "accumulation_steps": 1,
        "objective": "ce",
        "lr_scheduler": False,
        "discriminative_learning": True,
        "learning_rates": [0.1],
        "layer_lr_decay": 0.5,
        "weight_decay": 0.01,
        "train_highway": True,
        "train_gates": True,
        "early_exit_entropy": -1.0,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _assert_parameters_are_grouped(
    module: Union[LtDeeBert, LtBerxit], parameter_marker: str
) -> None:
    groups = module._get_optimizer_parameters()
    grouped_parameter_ids = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    expected_parameter_ids = {
        id(parameter)
        for name, parameter in module.named_parameters()
        if parameter_marker in name
    }

    assert expected_parameter_ids
    assert expected_parameter_ids <= grouped_parameter_ids


def test_deebert_shipped_config_includes_ramp_parameters(tmp_path: Path) -> None:
    model_config = _model_config(tmp_path)
    module = LtDeeBert(
        training_config=_ramp_training_config(),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=DeeBertModel(model_config),
    )

    _assert_parameters_are_grouped(module, ".ramp.")


def test_berxit_shipped_config_includes_ramp_parameters(tmp_path: Path) -> None:
    model_config = _model_config(tmp_path)
    module = LtBerxit(
        training_config=_ramp_training_config(),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=BerxitModel(model_config),
    )

    _assert_parameters_are_grouped(module, ".ramp.")


def test_berxit_non_discriminative_training_includes_gate_parameters(
    tmp_path: Path,
) -> None:
    model_config = _model_config(tmp_path)
    module = LtBerxit(
        training_config=_ramp_training_config(discriminative_learning=False),
        pretrained_model=str(tmp_path),
        num_labels=2,
        model=BerxitModel(model_config),
    )

    _assert_parameters_are_grouped(module, ".gates.")
