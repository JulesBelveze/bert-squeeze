import logging

import torch
from pkg_resources import resource_filename
from transformers import AutoConfig

from ...data import LrFeaturizer, TransformerFeaturizer
from ...distillation import Transformer2LR, Transformer2Transformer
from ...distillation.utils.labeler import HardLabeler
from ...models import LtCustomBert, LtCustomLabse
from ..utils_fct import load_model_from_exp


class DistillationArtifactsLoader:
    TEACHER_CLASSES = {"labse": LtCustomLabse, "bert": LtCustomBert}
    DISTIL_CLASSES = {"t2lr": Transformer2LR, "t2t": Transformer2Transformer}

    def __init__(self, config):
        self.config = config

        config.teacher.checkpoint_path = resource_filename(
            "bert-squeeze", config.teacher.checkpoint_path
        )
        self.fine_tuned_teacher = load_model_from_exp(
            config.teacher.checkpoint_path, self.teacher_class
        )

    @property
    def model_class(self):
        return self.DISTIL_CLASSES[self.config.task.strategy]

    @property
    def student_featurizer(self):
        return {
            "lr": LrFeaturizer(self.config.dataset, self.config.student.embed_dim),
            "transformer": TransformerFeaturizer(
                self.config.dataset,
                self.config.student.pretrained_tokenizer,
                self.config.student.max_seq_length,
            ),
        }[self.config.student.model]

    @property
    def teacher_class(self):
        return self.TEACHER_CLASSES[self.config.teacher.model_type]

    @property
    def teacher_model_config(self):
        return AutoConfig.from_pretrained(
            self.config.model["pretrained_model"],
            num_labels=int(self.config.model["num_labels"]),
        )

    @property
    def n_gpu(self):
        return torch.cuda.device_count()


class HardDistillationArtifactsLoader(DistillationArtifactsLoader):
    def __init__(self, config):
        super().__init__(config)
        self.labeler = HardLabeler(
            self.fine_tuned_teacher,
            config.hard_dataset,
            config.teacher.pretrained_tokenizer,
            config.teacher.max_seq_length,
        )

    def get_hard_labeled_data(self):
        logging.info("Starting to label samples using the fine tuned teacher.")
        ids_with_labels = self.labeler.label_dataset()
        logging.info("Dataset fully labeled.")
        return ids_with_labels
