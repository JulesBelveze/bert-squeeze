import logging
import os
import sys
from typing import List

from hydra.utils import get_class
from omegaconf import OmegaConf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_model_from_exp(path_to_folder: str, module):
    config_file = os.path.join(path_to_folder, ".hydra/config.yaml")
    config = OmegaConf.load(config_file)

    checkpoints = [file for file in os.listdir(os.path.join(path_to_folder, "checkpoints")) if file.endswith(".ckpt")]
    checkpoint_path = os.path.join(path_to_folder, "checkpoints", checkpoints[-1])

    logging.info(f"Loading model '{module}'")

    model = get_class(module).load_from_checkpoint(
        checkpoint_path,
        training_config=config,
        model_config=config.model.model_config,
        num_labels=config.model.num_labels
    )
    logging.info(f"Model '{module}' successfully loaded.")
    return model


def get_neptune_tags(args) -> List[str]:
    """Function returning Neptune experiment tags"""
    tags = []
    if args.task.name == "distil":
        tags.append("-".join([args.task.name, args.task.strategy]))
        tags.append(f"teacher-{args.model.teacher_config.name}")
        tags.append(f"student-{args.model.student_config.name}")
        tags.append(f"alpha-{args.train.alpha}")
        tags.append(args.data.student_module.dataset_config.name)
    else:
        if args.model.get("model_config", None) is not None:
            tags.append(args.model.model_config)
        else:
            tags.append(args.model.name)
        tags.append(args.data.dataset_config.name)
    tags.append(args.train.objective)
    tags.append(args.train.optimizer)
    return tags
