from __future__ import annotations

from typing import Union, cast

from overrides import overrides
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["GroupCompatibleReduceLROnPlateau"]


class GroupCompatibleReduceLROnPlateau(ReduceLROnPlateau):
    @overrides
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        migrated_state = dict(state_dict)
        migrated_state["min_lrs"] = self._migrated_min_lrs(state_dict.get("min_lrs"))
        migrated_state["_last_lr"] = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        super().load_state_dict(migrated_state)

    def _migrated_min_lrs(self, saved_min_lrs: object) -> list[float]:
        current_min_lrs = [float(value) for value in self.min_lrs]
        if not isinstance(saved_min_lrs, list) or not all(
            isinstance(value, (int, float)) for value in saved_min_lrs
        ):
            return current_min_lrs

        min_lrs = cast(list[Union[int, float]], saved_min_lrs)
        group_count = len(self.optimizer.param_groups)
        if len(min_lrs) == group_count:
            return [float(value) for value in min_lrs]
        if min_lrs and all(value == min_lrs[0] for value in min_lrs):
            return [float(min_lrs[0])] * group_count
        return current_min_lrs
