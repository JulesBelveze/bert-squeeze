import logging
import os
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch.nn
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from pkg_resources import resource_filename

from bert_squeeze.utils.utils_fct import deep_update, load_model_from_exp

CONFIG_MAPPER = {
    "distil": "distil.yaml",
    "distil-parallel": "distil_parallel.yaml",
    "distil-soft": "distil_soft.yaml",
    "distil-hard": "distil_hard.yaml",
}


class DistilAssistant(object):
    """
    Helper object that holds and instantiate the needed artifacts for distillation.

    It will load a default configuration to distil a teacher model into a student one.
    The configuration can be overwritten by passing keyword arguments.
    The configuration contains five main sub-configurations:
    - general: various high level parameters unrelated to the training procedure
    - train: training related parameters
    - model:
        * teacher: parameters necessary to build and define the teacher model
        * student: parameters necessary to build and define the student model
    - data: parameters necessary to define the dataset and featurize it

    Args:
        name (str):
            name of the base model to fine-tune
        general_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'general' configuration
        train_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'train' configuration
        student_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'model.student' configuration
        teacher_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'model.teacher' configuration
        data_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'data' configuration
        logger_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'logger' configuration
        callbacks (List[Callback]):
            list of callbacks to use during training

    Example:
        >>> from bert_squeeze.assistants import DistilAssistant
        >>> distil_assistant = DistilAssistant(
                "distil-parallel",
                data_kwargs={"path": resource_filename("bert_squeeze", "data/local_datasets/parallel_dataset.py")},
                teacher_kwargs={
                    "_target_": transformers.models.auto.AutoModelForSequenceClassification.from_pretrained
                    "pretrained_model_name_or_path": "bert-base-cased"
                }
            )
    """

    def __init__(
        self,
        name: str,
        general_kwargs: Dict[str, Any] = None,
        train_kwargs: Dict[str, Any] = None,
        student_kwargs: Dict[str, Any] = None,
        teacher_kwargs: Dict[str, Any] = {},
        data_kwargs: Dict[str, Any] = None,
        logger_kwargs: Dict[str, Any] = None,
        callbacks: List[Callback] = None,
    ):
        conf = OmegaConf.load(
            resource_filename(
                "bert_squeeze", os.path.join("assistants/configs", CONFIG_MAPPER[name])
            )
        )
        self._teacher_checkpoint = teacher_kwargs.pop("checkpoint_path", None)

        for name in ["teacher_module", "student_module"]:
            conf["data"][name]["dataset_config"] = deep_update(
                conf["data"][name]["dataset_config"], data_kwargs
            )

        for name, kws in zip(
            ["general", "train", "data", "logger", "callbacks"],
            [general_kwargs, train_kwargs, data_kwargs, logger_kwargs, callbacks],
        ):
            if kws is not None:
                conf[name] = deep_update(conf[name], kws)

        for name, kws in zip(["teacher", "student"], [teacher_kwargs, student_kwargs]):
            if kws is not None:
                conf["model"][name] = deep_update(conf["model"][name], kws)

        self.name = name
        self.general = conf["general"]
        self.train = conf["train"]

        self._model_conf = conf["model"]
        self._data_conf = conf["data"]
        self._logger_conf = conf.get("logger")
        self._callbacks_conf = conf.get("callbacks", [])

        self._model = None
        self._data = None
        self._logger = None
        self._callbacks = None

    @property
    def teacher_config(self) -> DictConfig:
        """"""
        return self._model_conf["teacher"]

    @property
    def student_config(self) -> DictConfig:
        """"""
        return self._model_conf["student"]

    @property
    def model(self) -> Any:
        """"""
        if self._model is None:
            self.model = instantiate(self._model_conf)

            if self._teacher_checkpoint is not None:
                if isinstance(self.teacher, pl.LightningModule):
                    self.model.teacher = load_model_from_exp(
                        self._teacher_checkpoint, module=self.model.teacher
                    )
                elif isinstance(self.teacher, torch.nn.Module):
                    self.model.teacher.load_state_dict(
                        torch.load(self._teacher_checkpoint)
                    )
                else:
                    raise TypeError(
                        f"Unexpected type '{type(self.teacher)}' for 'teacher'."
                    )

        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    @property
    def student(self) -> Any:
        """"""
        if self._model is None:
            logging.warning("The Distiller has not been instantiated.")
            return None
        return self.model.student

    @property
    def teacher(self) -> Any:
        """"""
        if self._model is None:
            logging.warning("The Distiller has not been instantiated.")
            return None
        return self.model.teacher

    @property
    def data(self) -> Any:
        """"""
        if self._data is None:
            data = instantiate(self._data_conf, _recursive_=True)
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
                self.callbacks = [
                    instantiate(callback) for callback in self._callbacks_conf
                ]
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
