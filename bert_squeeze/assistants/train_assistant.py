from copy import deepcopy
from importlib import resources
from typing import Dict, List, Optional

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from omegaconf import OmegaConf

from bert_squeeze.utils.utils_fct import deep_update

CONFIG_MAPPER = {
    "lr": "train_lr.yaml",
    "bert": "train_bert.yaml",
    "lstm": "train_lstm.yaml",
    "deebert": "train_deebert.yaml",
    "berxit": "train_berxit.yaml",
    "fastbert": "train_fastbert.yaml",
    "theseusbert": "train_theseus_bert.yaml",
    "adapter": "train_adapter.yaml",
    "t5": "train_t5.yaml",
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
        general_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'general' configuration
        train_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'train' configuration
        model_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'model' configuration
        data_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'data' configuration
        logger_kwargs (Dict[str, object]):
            keyword arguments that can be added or overwrite the default 'logger' configuration
        callbacks (List[Callback]):
            list of callbacks to use during training
    """

    def __init__(
        self,
        name: str,
        general_kwargs: Optional[Dict[str, object]] = None,
        train_kwargs: Optional[Dict[str, object]] = None,
        model_kwargs: Optional[Dict[str, object]] = None,
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
        data_overrides = deepcopy(data_kwargs) if data_kwargs is not None else None
        if data_overrides is not None:
            dataset_overrides = data_overrides.pop("dataset_config", None)
            if dataset_overrides is not None:
                conf["data"]["dataset_config"] = deep_update(
                    conf["data"]["dataset_config"], dataset_overrides
                )

        for section_name, overrides in zip(
            ["general", "train", "model", "data", "logger", "callbacks"],
            [
                general_kwargs,
                train_kwargs,
                model_kwargs,
                data_overrides,
                logger_kwargs,
                callbacks,
            ],
        ):
            if overrides is not None:
                base = conf.get(section_name)
                conf[section_name] = (
                    overrides if base is None else deep_update(base, overrides)
                )

        self.name = name
        self.general = conf["general"]
        self.train = conf["train"]
        self._model_conf = conf["model"]
        self._data_conf = conf["data"]
        self._logger_conf = conf.get("logger")
        self._callbacks_conf = conf.get("callbacks", [])

        self._model: Optional[pl.LightningModule] = None
        self._data: Optional[pl.LightningDataModule] = None
        self._logger = None
        self._callbacks = None

    @property
    def model(self) -> pl.LightningModule:
        """"""
        if self._model is None:
            self.model = instantiate(self._model_conf)
        return self._model

    @model.setter
    def model(self, value: pl.LightningModule) -> None:
        """"""
        self._model = value

    @property
    def data(self) -> pl.LightningDataModule:
        """"""
        if self._data is None:
            data = instantiate(self._data_conf)
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
