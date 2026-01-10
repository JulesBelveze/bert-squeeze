from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Optional

from omegaconf import DictConfig

_LOGGER_BACKENDS: Mapping[str, str] = {
    "aim": "lightning.pytorch.loggers.AimLogger",
    "tensorboard": "lightning.pytorch.loggers.TensorBoardLogger",
    "tb": "lightning.pytorch.loggers.TensorBoardLogger",
    "neptune": "lightning.pytorch.loggers.NeptuneLogger",
}


def resolve_logger_config(
    logger_config: Optional[Mapping[str, object]],
    *,
    default_save_dir: Optional[str] = None,
) -> Optional[MutableMapping[str, object]]:
    """
    Normalize logger configuration passed to assistants.

    The assistants historically expected a Hydra instantiation dict containing ``_target_``.
    This helper adds support for a lightweight preset syntax:

    .. code-block:: python

        logger_kwargs = {"backend": "aim", "repo": ".aim", "experiment": "exp-name"}

    Parameters
    ----------
    logger_config:
        Logger configuration dictionary passed by the user.
    default_save_dir:
        Fallback directory used by the TensorBoard preset when ``save_dir`` is omitted.
    """
    if logger_config is None:
        return None

    resolved: MutableMapping[str, object]
    if isinstance(logger_config, DictConfig):
        resolved = logger_config.copy()
    else:
        resolved = dict(logger_config)
    if "_target_" in resolved:
        return resolved

    backend = resolved.get("backend")
    if "backend" in resolved:
        del resolved["backend"]
    if backend is None:
        raise ValueError(
            "logger_kwargs must include either a Hydra '_target_' key or a 'backend' key "
            f"({sorted(_LOGGER_BACKENDS)})."
        )
    if not isinstance(backend, str):
        raise TypeError(
            f"logger_kwargs.backend must be a string, got {type(backend).__name__}."
        )

    backend_key = backend.strip().lower()
    target = _LOGGER_BACKENDS.get(backend_key)
    if target is None:
        raise ValueError(
            f"Unsupported logger backend '{backend}'. Supported backends: "
            f"{sorted(_LOGGER_BACKENDS)}."
        )
    resolved["_target_"] = target

    if backend_key in {"tensorboard", "tb"} and "save_dir" not in resolved:
        if default_save_dir is None:
            raise ValueError(
                "default_save_dir must be provided when using the tensorboard logger preset."
            )
        resolved["save_dir"] = default_save_dir

    return resolved
