from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Optional, TypedDict, Union, cast

from torch import nn
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict

__all__ = [
    "OptimizerParameterGroup",
    "build_optimizer_parameter_groups",
    "register_legacy_optimizer_state_migration",
]


class _RequiredOptimizerParameterGroup(TypedDict):
    params: list[nn.Parameter]
    weight_decay: float


class OptimizerParameterGroup(_RequiredOptimizerParameterGroup, total=False):
    lr: float


_LAYER_PATTERNS = (
    (re.compile(r"(?:^|\.)block\.(\d+)\."), False),
    (re.compile(r"(?:^|\.)layers\.(\d+)\."), False),
    (re.compile(r"(?:^|\.)h\.(\d+)\."), False),
    (re.compile(r"(?:^|\.)layer\.(\d+)\."), True),
)
_NO_DECAY_NAMES = ("bias", "gamma", "beta", "LayerNorm.weight", "layer_norm.weight")


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

    parameters_by_layer, remaining_parameters, uses_legacy_layout = (
        _split_parameters_by_layer(parameters)
    )
    layer_indices = sorted(parameters_by_layer)
    if not layer_indices:
        raise ValueError("No encoder layers found for discriminative learning.")

    layer_rates = _layer_rates(learning_rates, layer_lr_decay, len(layer_indices))
    layer_rate_by_index = dict(zip(layer_indices, layer_rates))
    preserve_legacy_slots = uses_legacy_layout and all(
        0 <= index < 12 for index in layer_indices
    )
    group_indices = list(range(12)) if preserve_legacy_slots else layer_indices
    groups = []
    for use_weight_decay in (True, False):
        for layer_index in group_indices:
            layer_parameters = [
                parameter
                for name, parameter in parameters_by_layer.get(layer_index, [])
                if _uses_weight_decay(name) == use_weight_decay
            ]
            if layer_parameters or preserve_legacy_slots:
                groups.append(
                    _parameter_group(
                        layer_parameters,
                        weight_decay,
                        use_weight_decay,
                        layer_rate_by_index.get(layer_index, layer_rates[-1]),
                    )
                )

    groups.extend(_weight_decay_groups(remaining_parameters, weight_decay))
    return groups


def register_legacy_optimizer_state_migration(
    optimizer: Optimizer,
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> None:
    legacy_parameter_groups = _legacy_parameter_groups(list(named_parameters))

    def migrate_state_dict(
        current_optimizer: Optimizer, state_dict: StateDict
    ) -> Optional[StateDict]:
        saved_groups = cast(list[dict[str, object]], state_dict["param_groups"])
        if len(saved_groups) != len(legacy_parameter_groups):
            return None

        saved_metadata_by_parameter = _saved_parameter_metadata(
            saved_groups, legacy_parameter_groups
        )
        if saved_metadata_by_parameter is None:
            return None

        current_groups = cast(list[dict[str, object]], current_optimizer.param_groups)
        serialized_groups = cast(
            list[dict[str, object]], current_optimizer.state_dict()["param_groups"]
        )
        migrated_groups = []
        for current_group, serialized_group in zip(current_groups, serialized_groups):
            current_parameters = cast(list[nn.Parameter], current_group["params"])
            if any(
                id(parameter) not in saved_metadata_by_parameter
                for parameter in current_parameters
            ):
                return None
            source_group_indices = {
                saved_metadata_by_parameter[id(parameter)][1]
                for parameter in current_parameters
            }
            if len(source_group_indices) == 1:
                source_group_index = next(iter(source_group_indices))
                source_group = saved_groups[source_group_index]
                migrated_group = {
                    key: value for key, value in source_group.items() if key != "params"
                }
            else:
                migrated_group = {
                    key: value
                    for key, value in serialized_group.items()
                    if key != "params"
                }
            migrated_group["params"] = [
                saved_metadata_by_parameter[id(parameter)][0]
                for parameter in current_parameters
            ]
            migrated_groups.append(migrated_group)

        migrated_state_dict = dict(state_dict)
        migrated_state_dict["param_groups"] = migrated_groups
        return cast(StateDict, migrated_state_dict)

    optimizer.register_load_state_dict_pre_hook(migrate_state_dict)


def _split_parameters_by_layer(
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> tuple[
    dict[int, list[tuple[str, nn.Parameter]]],
    list[tuple[str, nn.Parameter]],
    bool,
]:
    parameters_by_layer: dict[int, list[tuple[str, nn.Parameter]]] = {}
    remaining_parameters = []
    uses_legacy_layout = True
    for name, parameter in parameters:
        layer_match = _layer_match(name)
        if layer_match is None:
            remaining_parameters.append((name, parameter))
            continue
        layer_index, is_legacy_layer = layer_match
        uses_legacy_layout = uses_legacy_layout and is_legacy_layer
        parameters_by_layer.setdefault(layer_index, []).append((name, parameter))
    return parameters_by_layer, remaining_parameters, uses_legacy_layout


def _legacy_parameter_groups(
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> list[list[nn.Parameter]]:
    layer_keys = [f"layer.{index}." for index in range(12)]
    groups: list[list[nn.Parameter]] = []
    for use_weight_decay in (True, False):
        groups.extend(
            [
                parameter
                for name, parameter in parameters
                if layer_key in name and _uses_weight_decay(name) == use_weight_decay
            ]
            for layer_key in layer_keys
        )
    for use_weight_decay in (True, False):
        groups.append(
            [
                parameter
                for name, parameter in parameters
                if not any(layer_key in name for layer_key in layer_keys)
                and _uses_weight_decay(name) == use_weight_decay
            ]
        )
    return groups


def _saved_parameter_metadata(
    saved_groups: Sequence[dict[str, object]],
    legacy_parameter_groups: Sequence[list[nn.Parameter]],
) -> Optional[dict[int, tuple[int, int]]]:
    saved_metadata_by_parameter: dict[int, tuple[int, int]] = {}
    for group_index, (saved_group, legacy_parameters) in enumerate(
        zip(saved_groups, legacy_parameter_groups)
    ):
        saved_ids = cast(list[int], saved_group["params"])
        if len(saved_ids) != len(legacy_parameters):
            return None
        saved_metadata_by_parameter.update(
            (id(parameter), (saved_id, group_index))
            for parameter, saved_id in zip(legacy_parameters, saved_ids)
        )
    return saved_metadata_by_parameter


def _layer_match(parameter_name: str) -> Optional[tuple[int, bool]]:
    for pattern, is_legacy_layer in _LAYER_PATTERNS:
        match = pattern.search(parameter_name)
        if match is not None:
            return int(match.group(1)), is_legacy_layer
    return None


def _layer_rates(
    learning_rates: Union[float, Sequence[float]],
    layer_lr_decay: float,
    layer_count: int,
) -> list[float]:
    rates = (
        [float(rate) for rate in learning_rates]
        if isinstance(learning_rates, Sequence)
        else [float(learning_rates)]
    )
    if not rates:
        raise ValueError("At least one learning rate is required.")
    if len(rates) == 1:
        return [
            rates[0] * pow(layer_lr_decay, layer_count - index - 1)
            for index in range(layer_count)
        ]
    if len(rates) != layer_count:
        raise ValueError(
            f"Expected {layer_count} layer learning rates, received {len(rates)}."
        )
    return rates


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
        if not parameters:
            continue
        groups.append(
            _parameter_group(
                parameters,
                weight_decay,
                use_weight_decay,
                None,
            )
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
    return not any(no_decay in parameter_name for no_decay in _NO_DECAY_NAMES)
