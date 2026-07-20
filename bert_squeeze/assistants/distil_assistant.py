import logging
from copy import deepcopy
from importlib import resources
from typing import Dict, List, Optional, Union, cast

import lightning.pytorch as pl
import torch.nn
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from bert_squeeze.utils.utils_fct import deep_update, load_model_from_exp

CONFIG_MAPPER = {
    "distil": "distil.yaml",
    "distil-parallel": "distil_parallel.yaml",
    "distil-soft": "distil_soft.yaml",
    "distil-hard": "distil_hard.yaml",
    "distil-seq2seq": "distil_seq2seq.yaml",
}

DATA_SECTION_KEYS = {
    "_target_",
    "teacher_module",
    "student_module",
    "soft_data_config",
    "hard_labeler",
    "train_batch_size",
    "eval_batch_size",
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
        general_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'general' configuration
        train_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'train' configuration
        student_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'model.student' configuration
        teacher_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'model.teacher' configuration
        data_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'data' configuration
        logger_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'logger' configuration
        callbacks (List[Callback]):
            list of callbacks to use during training

    Example:
        >>> from importlib import resources
        >>> from bert_squeeze.assistants import DistilAssistant
        >>> distil_assistant = DistilAssistant(
                "distil-parallel",
                data_kwargs={
                    "path": str(
                        resources.files("bert_squeeze").joinpath(
                            "data/local_datasets/parallel_dataset.py"
                        )
                    )
                },
                teacher_kwargs={
                    "_target_": transformers.models.auto.AutoModelForSequenceClassification.from_pretrained
                    "pretrained_model_name_or_path": "bert-base-cased"
                }
            )
    """

    def __init__(
        self,
        name: str,
        general_kwargs: Optional[Dict[str, object]] = None,
        train_kwargs: Optional[Dict[str, object]] = None,
        student_kwargs: Optional[Dict[str, object]] = None,
        teacher_kwargs: Optional[Dict[str, object]] = None,
        data_kwargs: Optional[Dict[str, object]] = None,
        logger_kwargs: Optional[Dict[str, object]] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        try:
            config_name = CONFIG_MAPPER[name]
        except KeyError:
            raise ValueError(
                f"'{name}' is not a valid configuration name, please use one of the"
                f" following: {CONFIG_MAPPER.keys()}"
            )

        config_path = resources.files("bert_squeeze").joinpath(
            "assistants/configs", config_name
        )
        with resources.as_file(config_path) as resolved_path:
            conf = OmegaConf.load(resolved_path)
        teacher_overrides = deepcopy(teacher_kwargs) if teacher_kwargs is not None else {}
        data_overrides = deepcopy(data_kwargs) if data_kwargs is not None else None
        self._teacher_checkpoint = teacher_overrides.pop("checkpoint_path", None)

        if data_overrides is not None:
            shared_dataset_overrides = cast(
                Dict[str, object], data_overrides.pop("dataset_config", {})
            )
            shared_dataset_overrides.update(
                {
                    key: value
                    for key, value in data_overrides.items()
                    if key not in DATA_SECTION_KEYS
                }
            )

            for module_name in ["teacher_module", "student_module"]:
                conf["data"][module_name]["dataset_config"] = deep_update(
                    conf["data"][module_name]["dataset_config"],
                    shared_dataset_overrides,
                )

        for section_name, overrides in zip(
            ["general", "train", "data", "logger", "callbacks"],
            [general_kwargs, train_kwargs, data_overrides, logger_kwargs, callbacks],
        ):
            if overrides is not None:
                base = conf.get(section_name)
                if base is None:
                    conf[section_name] = overrides
                    continue

                if (
                    isinstance(overrides, dict)
                    and "_target_" in overrides
                    and overrides["_target_"] != conf[section_name]["_target_"]
                ):
                    del conf[section_name]
                    conf[section_name] = overrides
                elif section_name == "data":
                    for module_name in ["teacher_module", "student_module"]:
                        if (
                            module_name in overrides
                            and "_target_" in overrides[module_name]
                            and conf[section_name][module_name]["_target_"]
                            != overrides[module_name]["_target_"]
                        ):
                            del conf[section_name][module_name]
                            conf[section_name][module_name] = overrides[module_name]

                conf[section_name] = deep_update(conf[section_name], overrides)

        for role, overrides in zip(
            ["teacher", "student"], [teacher_overrides, student_kwargs]
        ):
            if overrides is not None:
                conf["model"][role] = deep_update(conf["model"][role], overrides)

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
    def model(self) -> pl.LightningModule:
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
    def model(self, value: pl.LightningModule) -> None:
        self._model = value

    @property
    def student(self) -> Optional[Union[pl.LightningModule, torch.nn.Module]]:
        """"""
        if self._model is None:
            logging.warning("The Distiller has not been instantiated.")
            return None
        return self.model.student

    @property
    def teacher(self) -> Optional[Union[pl.LightningModule, torch.nn.Module]]:
        """"""
        if self._model is None:
            logging.warning("The Distiller has not been instantiated.")
            return None
        return self.model.teacher

    @property
    def data(self) -> pl.LightningDataModule:
        """"""
        if self._data is None:
            data = instantiate(self._data_conf, _recursive_=True)
            data.prepare_data()
            data.setup()
            self.data = data
        return self._data

    @data.setter
    def data(self, value: pl.LightningDataModule) -> None:
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
    def callbacks(self, value: Optional[List[Callback]]) -> None:
        """"""
        self._callbacks = value

    def __repr__(self):
        return f"<DistilAssistant(name={self.name})>"

    def __str__(self):
        return f"DistilAssistant_{self.name}"
