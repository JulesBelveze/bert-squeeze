from typing import Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from bert_squeeze.models.base_lt_module import BaseSeq2SeqTransformerModule
from bert_squeeze.utils.scorers import Scorer


class SimpleT5Model(BaseSeq2SeqTransformerModule):
    """
    Simple wrapper around a T5 model

    Args:
        training_config (DictConfig):
            training configuration
        pretrained_model (str):
            name of the pretrained model to use a backbone
        task (str):
            name of the task to perform
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        generate_kws (DictConfig):
             additional keywords to feed to the `.generate` method
    """

    BASE_CLASS_MODEL = T5ForConditionalGeneration

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        task: str,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        generate_kwargs: DictConfig = None,
        **kwargs,
    ):
        super().__init__(training_config, pretrained_model, task, model, scorer, **kwargs)
        self.generate_kwargs = generate_kwargs

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> Seq2SeqLMOutput:
        """
        Args:
            input_ids (torch.Tensor):
                indices of input sequence tokens in the vocabulary
            attention_mask (torch.Tensor):
                mask to avoid performing attention on padding token indices
            labels (torch.Tensor):
                labels to predict
        Returns:
            Seq2SeqLMOutput
        """
        model_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return model_outputs

    def training_step(self, batch, batch_idx, *args, **kwargs):
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        outputs = self.forward(**inputs)

        self.scorer.add(outputs.loss.detach())

        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            self.logger.experiment["train/loss"].log(
                value=np.mean(self.scorer.losses), step=self.global_step
            )

            self.logger.experiment["train/perplexity"].log(
                self.scorer.perplexity, step=self.global_step
            )
            self.scorer.reset()

        return {"loss": outputs.loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        outputs = self.forward(**inputs)
        prediction = self.model.generate(batch["input_ids"], **self.generate_kwargs)

        self.valid_scorer.add(
            loss=outputs.loss.detach(),
            predicted_tokens=prediction,
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )

        return {"loss": outputs.loss}
