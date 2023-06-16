import logging
import os
from typing import Any, Dict, List

from hydra.utils import instantiate
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from pydantic.utils import deep_update

CONFIG_MAPPER = {
    "lr": "train_lr.yaml",
    "bert": "train_bert.yaml",
    "lstm": "train_lstm.yaml",
    "deebert": "train_deebert.yaml",
    "fastbert": "train_fastbert.yaml",
    "theseusbert": "train_theseus_bert.yaml",
}


class TrainAssistant(object):
    """
    Helper object that holds and instantiate the needed for training.

    For every available model for fine-tuning it will load a default configuration that
    can be overwritten by passing some keyword arguments.
    It contains four main sub-configurations:

    - *general*: various high level parameters unrelated to the training procedure
    - *train*: training related parameters
    - *model*: parameters necessary to build and define the model
    - *data*: parameters necessary to define the dataset and featurize it

    Args:
        name (str):
            name of the base model to fine-tune
        general_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'general' configuration
        train_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'train' configuration
        model_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'model' configuration
        data_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'data' configuration
        logger_kwargs (Dict[str, Any]):
            keyword arguments that can be added or overwrite the default 'logger' configuration
        callbacks (List[Callback]):
            list of callbacks to use during training
    """

    def __init__(
        self,
        name: str,
        general_kwargs: Dict[str, Any] = None,
        train_kwargs: Dict[str, Any] = None,
        model_kwargs: Dict[str, Any] = None,
        data_kwargs: Dict[str, Any] = None,
        logger_kwargs: Dict[str, Any] = None,
        callbacks: List[Callback] = None,
    ):
        try:
            conf = OmegaConf.load(
                resource_filename(
                    "bert_squeeze",
                    os.path.join("assistants/configs", CONFIG_MAPPER[name]),
                )
            )
        except KeyError:
            raise ValueError(
                f"'{name}' is not a valid configuration name, please use one of the following: {CONFIG_MAPPER.keys()}"
            )
        if (
            data_kwargs is not None
            and data_kwargs.get("dataset_config", {}).get("path") is not None
        ):
            logging.warning(
                "Found value for `dataset_config.path` which conflicts with parameter `dataset_path`, using"
                "value from the later."
            )

        conf["data"]["dataset_config"] = deep_update(
            conf["data"]["dataset_config"], data_kwargs["dataset_config"]
        )
        del data_kwargs["dataset_config"]

        for name, kws in zip(
            ["general", "train", "model", "data", "logger", "callbacks"],
            [
                general_kwargs,
                train_kwargs,
                model_kwargs,
                data_kwargs,
                logger_kwargs,
                callbacks,
            ],
        ):
            if kws is not None:
                conf[name] = deep_update(conf[name], kws)

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
    def model(self) -> Any:
        """"""
        if self._model is None:
            self.model = instantiate(self._model_conf)
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        """"""
        self._model = value

    @property
    def data(self) -> Any:
        """"""
        if self._data is None:
            data = instantiate(self._data_conf)
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
    def callbacks(self) -> List[Callback]:
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
    def callbacks(self, value: List[Callback]) -> None:
        """"""
        self._callbacks = value

    def __repr__(self):
        return f"<TrainAssistant(name={self.name})>"

    def __str__(self):
        return f"TrainAssistant_{self.name}"
