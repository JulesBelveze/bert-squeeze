import torch
from transformers import AutoConfig

from ...models import DeeBert, LtCustomBert, LtCustomLabse


class TransformerArtifactsLoader:
    MODEL_CLASSES = {"labse": LtCustomLabse, "bert": LtCustomBert, "deebert": DeeBert}

    def __init__(self, config):
        self.config = config

    @property
    def model_config(self):
        return AutoConfig.from_pretrained(
            self.config.model["pretrained_model"],
            num_labels=int(self.config.model["num_labels"]),
        )

    @property
    def model_class(self):
        return self.MODEL_CLASSES[self.config.model["model_type"]]

    @property
    def n_gpu(self):
        return torch.cuda.device_count()
