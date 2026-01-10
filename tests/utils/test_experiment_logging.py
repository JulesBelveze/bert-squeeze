import sys
import types

from bert_squeeze.utils.experiment_logging import ExperimentLogger


class _DummyTensorboardExperiment:
    def __init__(self):
        self.text_calls = []
        self.figure_calls = []

    def add_text(self, name, text, global_step=None):
        self.text_calls.append((name, text, global_step))

    def add_figure(self, name, figure, global_step=None):
        self.figure_calls.append((name, figure, global_step))


class _DummyLogger:
    def __init__(self, experiment):
        self.experiment = experiment


def test_log_experiment_text_tensorboard_experiment():
    experiment = _DummyTensorboardExperiment()
    logger = _DummyLogger(experiment=experiment)

    ExperimentLogger(logger, step=12).add_text("eval/report", "hello")

    assert experiment.text_calls == [("eval/report", "hello", 12)]


def test_log_experiment_figure_tensorboard_experiment():
    experiment = _DummyTensorboardExperiment()
    logger = _DummyLogger(experiment=experiment)

    fig = object()
    ExperimentLogger(logger, step=3).add_figure("eval/fig", fig)

    assert experiment.figure_calls == [("eval/fig", fig, 3)]


def test_log_experiment_text_aim_run(monkeypatch):
    aim_module = types.ModuleType("aim")

    class Text:
        def __init__(self, value):
            self.value = value

    class Run:
        def __init__(self):
            self.calls = []

        def track(self, value, name=None, step=None):
            self.calls.append((value, name, step))

    aim_module.Run = Run
    aim_module.Text = Text
    monkeypatch.setitem(sys.modules, "aim", aim_module)

    run = Run()
    logger = _DummyLogger(experiment=run)

    ExperimentLogger(logger, step=7, epoch=2).add_text("eval/report", "hello")

    assert len(run.calls) == 1
    value, name, step = run.calls[0]
    assert isinstance(value, Text)
    assert value.value == "hello"
    assert name == "eval/report"
    assert step == 7


def test_log_experiment_figure_aim_run(monkeypatch):
    aim_module = types.ModuleType("aim")

    class Figure:
        def __init__(self, fig):
            self.fig = fig

    class Run:
        def __init__(self):
            self.calls = []

        def track(self, value, name=None, epoch=None):
            self.calls.append((value, name, epoch))

    aim_module.Run = Run
    aim_module.Figure = Figure
    monkeypatch.setitem(sys.modules, "aim", aim_module)

    run = Run()
    logger = _DummyLogger(experiment=run)

    fig = object()
    ExperimentLogger(logger, step=11, epoch=4).add_figure("eval/fig", fig)

    assert len(run.calls) == 1
    value, name, epoch = run.calls[0]
    assert isinstance(value, Figure)
    assert value.fig is fig
    assert name == "eval/fig"
    assert epoch == 4
