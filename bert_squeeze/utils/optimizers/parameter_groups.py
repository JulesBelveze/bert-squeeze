from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Optional, TypedDict, Union

from torch import nn

__all__ = ["OptimizerParameterGroup", "build_optimizer_parameter_groups"]


class _RequiredOptimizerParameterGroup(TypedDict):
    params: list[nn.Parameter]
    weight_decay: float


class OptimizerParameterGroup(_RequiredOptimizerParameterGroup, total=False):
    lr: float


_LayerKey = tuple[str, int]

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)(block)\.(\d+)\."),
    re.compile(r"(?:^|\.)(layers)\.(\d+)\."),
    re.compile(r"(?:^|\.)(h)\.(\d+)\."),
    re.compile(r"(?:^|\.)(layer)\.(\d+)\."),
)
_EMBEDDING_MODULES = (
    "embeddings",
    "embed_tokens",
    "embed_positions",
    "wte",
    "wpe",
)


def build_optimizer_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    discriminative_learning: bool,
    learning_rates: Union[float, Sequence[float]],
    layer_lr_decay: float,
    weight_decay: float,
) -> list[OptimizerParameterGroup]:
    parameters = list(named_parameters)
    if not discriminative_learning:
        return _weight_decay_groups(parameters, weight_decay)

    rates = _learning_rate_values(learning_rates)
    layer_keys = [_layer_key(name) for name, _ in parameters]
    stack_indices = _stack_indices(layer_keys)
    if not stack_indices:
        raise ValueError("No encoder layers found for discriminative learning.")

    layer_rate_by_key = _layer_rates_by_key(rates, layer_lr_decay, stack_indices)
    embedding_rate = min(layer_rate_by_key.values()) * layer_lr_decay
    head_rate = rates[-1]

    grouped_parameters: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for (name, parameter), layer_key in zip(parameters, layer_keys):
        learning_rate = _parameter_learning_rate(
            name,
            layer_key,
            layer_rate_by_key,
            embedding_rate,
            head_rate,
        )
        group_key = (learning_rate, _uses_weight_decay(name))
        grouped_parameters.setdefault(group_key, []).append(parameter)

    return [
        _parameter_group(parameters, weight_decay, use_weight_decay, learning_rate)
        for (learning_rate, use_weight_decay), parameters in grouped_parameters.items()
    ]


def _learning_rate_values(
    learning_rates: Union[float, Sequence[float]],
) -> list[float]:
    rates = (
        [float(rate) for rate in learning_rates]
        if isinstance(learning_rates, Sequence)
        else [float(learning_rates)]
    )
    if not rates:
        raise ValueError("At least one learning rate is required.")
    return rates


def _stack_indices(layer_keys: Sequence[Optional[_LayerKey]]) -> dict[str, set[int]]:
    indices: dict[str, set[int]] = {}
    for layer_key in layer_keys:
        if layer_key is None:
            continue
        stack, layer_index = layer_key
        indices.setdefault(stack, set()).add(layer_index)
    return indices


def _layer_rates_by_key(
    rates: Sequence[float],
    layer_lr_decay: float,
    stack_indices: dict[str, set[int]],
) -> dict[_LayerKey, float]:
    if layer_lr_decay <= 0 or layer_lr_decay > 1:
        raise ValueError("layer_lr_decay must be in (0, 1].")
    max_layer_count = max(len(indices) for indices in stack_indices.values())
    if len(rates) > 1 and len(rates) != max_layer_count:
        raise ValueError(
            f"Expected {max_layer_count} layer learning rates, received {len(rates)}."
        )

    rate_by_key: dict[_LayerKey, float] = {}
    for stack, indices in stack_indices.items():
        sorted_indices = sorted(indices)
        stack_rates = _stack_rates(rates, layer_lr_decay, len(sorted_indices))
        rate_by_key.update(
            ((stack, layer_index), rate)
            for layer_index, rate in zip(sorted_indices, stack_rates)
        )
    return rate_by_key


def _stack_rates(
    rates: Sequence[float], layer_lr_decay: float, layer_count: int
) -> list[float]:
    if len(rates) > 1:
        return list(rates[-layer_count:])
    return [
        rates[0] * pow(layer_lr_decay, layer_count - index - 1)
        for index in range(layer_count)
    ]


def _parameter_learning_rate(
    name: str,
    layer_key: Optional[_LayerKey],
    layer_rate_by_key: dict[_LayerKey, float],
    embedding_rate: float,
    head_rate: float,
) -> float:
    if layer_key is not None:
        return layer_rate_by_key[layer_key]
    if _is_embedding_parameter(name):
        return embedding_rate
    return head_rate


def _layer_key(parameter_name: str) -> Optional[_LayerKey]:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(parameter_name)
        if match is not None:
            stack = parameter_name[: match.start(1)] + match.group(1)
            return stack, int(match.group(2))
    return None


def _is_embedding_parameter(parameter_name: str) -> bool:
    parts = parameter_name.lower().split(".")
    return parts[-2:] == ["shared", "weight"] or any(
        module in _EMBEDDING_MODULES for module in parts[:-1]
    )


def _weight_decay_groups(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    weight_decay: float,
) -> list[OptimizerParameterGroup]:
    groups = []
    for use_weight_decay in (True, False):
        parameters = [
            parameter
            for name, parameter in named_parameters
            if _uses_weight_decay(name) == use_weight_decay
        ]
        if parameters:
            groups.append(
                _parameter_group(parameters, weight_decay, use_weight_decay, None)
            )
    return groups


def _parameter_group(
    parameters: list[nn.Parameter],
    weight_decay: float,
    use_weight_decay: bool,
    learning_rate: Optional[float],
) -> OptimizerParameterGroup:
    group = OptimizerParameterGroup(
        params=parameters,
        weight_decay=weight_decay if use_weight_decay else 0.0,
    )
    if learning_rate is not None:
        group["lr"] = learning_rate
    return group


def _uses_weight_decay(parameter_name: str) -> bool:
    parts = parameter_name.lower().split(".")
    parameter = parts[-1]
    if parameter in {"bias", "beta", "gamma"}:
        return False
    return parameter != "weight" or not any(
        "norm" in module or module.startswith("ln_") for module in parts[:-1]
    )
