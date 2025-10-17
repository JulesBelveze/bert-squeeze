import lightning.pytorch as pl
from omegaconf import DictConfig
from peft import get_peft_model
from transformerrs import AutoModelForSeq2SeqLM


class LLM(pl.LightningModule):
    """"""

    BASE_CLASS_MODEL = AutoModelForSeq2SeqLM

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        model,
        task: str,
        adapter_config,
        **kwargs,
    ):
        super().__init__()

        self.config = training_config

        self.pretrained_model = pretrained_model
        self.adapter_config = adapter_config

        unoptimized_model = (
            self.BASE_CLASS_MODEL.from_pretrained(pretrained_model)
            if model is None
            else model
        )

        self.model = get_peft_model(get_peft_model(unoptimized_model, adapter_config))
