import logging
from collections.abc import Callable, Iterator
from typing import Optional

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Small adapter around a Lightning logger to log non-metric artifacts across multiple backends
    Currently supported:
    - TensorBoard
    - Aim
    """

    def __init__(
        self,
        lightning_logger: Optional[object],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        self._logger = lightning_logger
        self._step = step
        self._epoch = epoch

    @classmethod
    def from_module(cls, module: object) -> "ExperimentLogger":
        return cls(
            getattr(module, "logger", None),
            step=getattr(module, "global_step", None),
            epoch=getattr(module, "current_epoch", None),
        )

    def add_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        resolved_step, resolved_epoch = self._resolve_step_and_epoch(
            step=step, epoch=epoch
        )
        log_experiment_text(
            self._logger,
            name=name,
            text=text,
            step=resolved_step,
            epoch=resolved_epoch,
        )

    def add_figure(
        self,
        name: str,
        figure: object,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        resolved_step, resolved_epoch = self._resolve_step_and_epoch(
            step=step, epoch=epoch
        )
        log_experiment_figure(
            self._logger,
            name=name,
            figure=figure,
            step=resolved_step,
            epoch=resolved_epoch,
        )

    def _resolve_step_and_epoch(
        self, *, step: Optional[int], epoch: Optional[int]
    ) -> tuple[Optional[int], Optional[int]]:
        resolved_step = self._step if step is None else step
        resolved_epoch = self._epoch if epoch is None else epoch
        return resolved_step, resolved_epoch


def log_experiment_text(
    lightning_logger: Optional[object],
    name: str,
    text: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
) -> None:
    """
    Log text to the underlying experiment, if supported.

    Supports TensorBoard's SummaryWriter (``add_text``) and Aim (``Run.track(Text)``).
    """
    _log_experiment_artifact(
        lightning_logger,
        name=name,
        value=text,
        step=step,
        epoch=epoch,
        tensorboard_method="add_text",
        aim_wrapper="Text",
    )


def log_experiment_figure(
    lightning_logger: Optional[object],
    name: str,
    figure: object,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
) -> None:
    """
    Log a matplotlib figure to the underlying experiment, if supported.

    Supports TensorBoard's SummaryWriter (``add_figure``) and Aim (``Run.track(Figure)``).
    """
    _log_experiment_artifact(
        lightning_logger,
        name=name,
        value=figure,
        step=step,
        epoch=epoch,
        tensorboard_method="add_figure",
        aim_wrapper="Figure",
    )


def _log_experiment_artifact(
    lightning_logger: Optional[object],
    *,
    name: str,
    value: object,
    step: Optional[int],
    epoch: Optional[int],
    tensorboard_method: str,
    aim_wrapper: str,
) -> None:
    for experiment in _iter_experiments(lightning_logger):
        if _try_tensorboard_add(
            experiment, method=tensorboard_method, name=name, value=value, step=step
        ):
            continue
        _try_aim_track(
            experiment,
            wrapper=aim_wrapper,
            value=value,
            name=name,
            step=step,
            epoch=epoch,
        )


def _iter_experiments(lightning_logger: Optional[object]) -> Iterator[object]:
    if lightning_logger is None:
        return

    if isinstance(lightning_logger, (list, tuple, set)):
        for sublogger in lightning_logger:
            yield from _iter_experiments(sublogger)
        return

    loggers = getattr(lightning_logger, "loggers", None)
    if loggers is not None:
        for sublogger in loggers:
            yield from _iter_experiments(sublogger)
        return

    experiment = getattr(lightning_logger, "experiment", None)
    yield experiment if experiment is not None else lightning_logger


def _try_tensorboard_add(
    experiment: object, *, method: str, name: str, value: object, step: Optional[int]
) -> bool:
    fn = getattr(experiment, method, None)
    if fn is None or not callable(fn):
        return False

    try:
        fn(name, value, global_step=step)
    except TypeError:
        fn(name, value)
    return True


def _try_aim_track(
    experiment: object,
    *,
    wrapper: str,
    value: object,
    name: str,
    step: Optional[int],
    epoch: Optional[int],
) -> bool:
    aim_module = _import_aim()
    if aim_module is None:
        return False
    if not _looks_like_aim_run(experiment, aim_module):
        return False

    aim_object = _get_aim_object_constructor(aim_module, wrapper)
    wrapped_value = aim_object(value) if aim_object is not None else value

    _aim_track(experiment, value=wrapped_value, name=name, step=step, epoch=epoch)
    return True


def _looks_like_aim_run(experiment: object, aim_module: object) -> bool:
    track = getattr(experiment, "track", None)
    if experiment is None or track is None or not callable(track):
        return False

    run_type = getattr(aim_module, "Run", None)
    if isinstance(run_type, type) and isinstance(experiment, run_type):
        return True

    return getattr(type(experiment), "__module__", "").startswith("aim")


def _import_aim() -> Optional[object]:
    try:
        import aim
    except ModuleNotFoundError:
        return None
    except Exception:
        logger.debug("Failed to import aim", exc_info=True)
        return None
    return aim


def _get_aim_object_constructor(
    aim_module: object, wrapper_name: str
) -> Optional[Callable[[object], object]]:
    constructor = getattr(aim_module, wrapper_name, None)
    if constructor is not None and callable(constructor):
        return constructor

    try:
        from aim.sdk.objects import Figure, Text
    except Exception:
        return None
    return {"Text": Text, "Figure": Figure}.get(wrapper_name)


def _aim_track(
    experiment: object,
    *,
    value: object,
    name: str,
    step: Optional[int],
    epoch: Optional[int],
) -> None:
    track = getattr(experiment, "track", None)
    if track is None or not callable(track):
        return

    for call_kwargs in _iter_aim_track_kwargs(name=name, step=step, epoch=epoch):
        try:
            track(value, **call_kwargs)
            return
        except TypeError:
            continue
        except Exception:
            logger.debug("Aim tracking failed", exc_info=True)
            return


def _iter_aim_track_kwargs(
    *, name: str, step: Optional[int], epoch: Optional[int]
) -> Iterator[dict[str, object]]:
    """
    Yield Aim ``track`` kwargs in a best-effort order.

    Aim has changed accepted arguments across versions; try the most informative kwargs
    first, then progressively drop unsupported ones.
    """
    base: dict[str, object] = {"name": name}

    if step is not None and epoch is not None:
        yield {"name": name, "step": step, "epoch": epoch}
        yield {"name": name, "step": step}
        yield {"name": name, "epoch": epoch}
        yield base
        return

    if step is not None:
        yield {"name": name, "step": step}
        yield base
        return

    if epoch is not None:
        yield {"name": name, "epoch": epoch}
        yield base
        return

    yield base
