import collections.abc
import logging
import os
import sys
from typing import List

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_model_from_exp(
    path_to_folder: str, module: pl.LightningModule
) -> pl.LightningModule:
    """
    Helper function to load a `pl.LightningModule` from a previous experiment.
    The folder needs to have the following structure:

    .. code-block::

        folder
        ┣━━ .hydra/
        ┃   ┗━━ config.yaml
        ┗━━ checkpoints/
            ┣━━ checkpoints_0.ckpt
            ┃   ....
            ┗━━ checkpoints_n.ckpt

    Args:
        path_to_folder (str):
            path to folder containing the checkpoints and the Hydra configuration from
            a previous experiment.
        module (pl.LightningModule):
            Module to load the checkpoints from.

    Returns:
        pl.LightningModule: module resumed from the previous experiment.
    """
    config_file = os.path.join(path_to_folder, ".hydra/config.yaml")
    config = OmegaConf.load(config_file)

    checkpoints = [
        file
        for file in os.listdir(os.path.join(path_to_folder, "checkpoints"))
        if file.endswith(".ckpt")
    ]
    checkpoint_path = os.path.join(path_to_folder, "checkpoints", checkpoints[-1])

    logging.info(f"Loading model '{module}'")

    model = module.load_from_checkpoint(checkpoint_path, **config)
    logging.info(f"Model '{model}' successfully loaded.")
    return model


def get_neptune_tags(args: DictConfig) -> List[str]:
    """
    Function returning Neptune experiment tags from a Hydra configuration in order
    to filter sort experiments in the Neptune UI.

    Args:
        args (DictConfig):
            Configuration used to train the model.
    Returns:
        List[str]: list of tags that will be logged to the "tags" column of a Neptune
                   experiment.
    """
    tags = []
    if args.task.name == "distil":
        tags.append("-".join([args.task.name, args.task.strategy]))
        tags.append(f"teacher-{args.model.teacher_config.name}")
        tags.append(f"student-{args.model.student_config.name}")
        tags.append(f"alpha-{args.train.alpha}")
        tags.append(args.data.student_module.dataset_config.name)
    else:
        if args.model.get("pretrained_model", None) is not None:
            tags.append(args.model.pretrained_model)
        else:
            tags.append(args.model.name)
        tags.append(args.data.dataset_config.name)

    if args.train.get("objective", None):
        tags.append(args.train.objective)
    tags.append(args.train.optimizer)
    return tags


def deep_update(d, u):
    """"""
    if isinstance(u, list):
        if d is None:
            return u
        d.extend(u)
        return d

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
