import os
from typing import Any, Dict, List, Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from pydantic.utils import deep_update
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger

CONFIG_MAPPER = {
    "distil-parallel": "distil_parallel.yaml"
}


class DistilAssistant(object):
    """

    """

    def __init__(
            self,
            name: str,
            dataset_path: str,
            general_kwargs: Dict[str, Any] = None,
            train_kwargs: Dict[str, Any] = None,
            student_kwargs: Dict[str, Any] = None,
            teacher_kwargs: Dict[str, Any] = None,
            data_kwargs: Dict[str, Any] = None,
            logger_kwargs: Dict[str, Any] = None,
            callbacks: List[Callback] = None
    ):
        conf = OmegaConf.load(
            resource_filename("bert_squeeze", os.path.join("assistants/configs", CONFIG_MAPPER[name]))
        )

        for name in ["teacher_module", "student_module"]:
            conf["data"][name]["dataset_config"]["path"] = dataset_path

        for name, kws in zip(["general", "train", "student", "teacher", "data", "logger", "callbacks"],
                             [general_kwargs, train_kwargs, student_kwargs, teacher_kwargs, data_kwargs, logger_kwargs,
                              callbacks]):
            if kws is not None:
                conf[name] = deep_update(conf[name], kws)

        self.name = name
        self.general = conf["general"]
        self.train = conf["train"]
        self._student_conf = conf["model"]["student_config"]
        self._teacher_conf = conf["model"]["teacher_config"]
        self._data_conf = conf["data"]
        self._logger_conf = conf.get("logger")
        self._callbacks_conf = conf.get("callbacks", [])

        self._teacher = None
        self._student = None
        self._data = None
        self._logger = None
        self._callbacks = None

    @property
    def student(self) -> Any:
        """"""
        if self._student is None:
            self.student = instantiate(self._student_conf, recurse=True)
        return self._student

    @student.setter
    def student(self, value: Any) -> None:
        """"""
        self._student = value

    @property
    def teacher(self) -> Any:
        """"""
        if self._teacher is None:
            self.teacher = instantiate(self._teacher_conf, recurse=True)
        return self._teacher

    @teacher.setter
    def teacher(self, value: Any) -> None:
        """"""
        self._teacher = value

    @property
    def data(self) -> Any:
        """"""
        if self._data is None:
            data = instantiate(self._data_conf, recurse=True)
            data.prepare_data()
            data.setup()
            self.data = data
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """"""
        self._data = value

    @property
    def logger(self) -> Logger:
        """"""
        if self._logger is None:
            if self._logger_conf is not None:
                self.logger = instantiate(self._logger_conf)
            else:
                self.logger = TensorBoardLogger(self.general["output_dir"])
        return self._logger

    @logger.setter
    def logger(self, value: Logger) -> None:
        """"""
        self._logger = value

    @property
    def callbacks(self) -> Optional[List[Callback]]:
        """"""
        if self._callbacks is None:
            if self._callbacks_conf is not None:
                self.callbacks = [instantiate(callback) for callback in self._callbacks_conf]
            else:
                self.callbacks = []
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value) -> None:
        """"""
        self._callbacks = value

    def __repr__(self):
        return f"<DistilAssistant(name={self.name})>"

    def __str__(self):
        return f"DistilAssistant_{self.name}"
