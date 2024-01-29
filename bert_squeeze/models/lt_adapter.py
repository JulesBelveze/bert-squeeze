from typing import List, Tuple, Union

import torch
from adapters import AutoAdapterModel
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoConfig

from .base_lt_module import BaseSequenceClassificationTransformerModule


class LtAdapter(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune adapters for Transformer-based language models on sequence classification task.

    Note: It uses the adapters library under the hood, check it out to better understand how to choose parameters
    https://github.com/adapter-hub/adapters

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels for the classification tasks
        pretrained_model (str):
            name of the pretrained Transformer model to use
        task_name (str):
            name for the adapter configuration
        adapter_config_name (str):
            nam of the adapter config to use
        labels (Union[List[str], List[int]]):
            list of labels used for the classification head
    """

    def __init__(
            self,
            training_config: DictConfig,
            num_labels: int,
            pretrained_model: str,
            task_name: str,
            adapter_config_name: str,
            labels: Union[List[str], List[int]],
            **kwargs,
    ):
        super().__init__(training_config, pretrained_model, num_labels, **kwargs)

        assert len(labels) == self.model_config.num_labels

        self.task_name = task_name
        self.adapter_config_name = adapter_config_name
        self.labels = labels

        self._build_model()

    @overrides
    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input_ids (torch.Tensor):
                sentence or sentences represented as tokens
            attention_mask (torch.Tensor):
                tells the model which tokens in the input_ids are words and which are padding.
                1 indicates a token and 0 indicates padding.
            token_type_ids (torch.Tensor):
                used when there are two sentences that need to be part of the input. It indicates which
                tokens are part of sentence1 and which are part of sentence2.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: logits obtained from model pass

        For specifications about model output, please refer to:
        https://github.com/huggingface/transformers/blob/b01f451ca38695c60175b34d245997ef4d18231d/src/transformers/modeling_outputs.py#L153
        """
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return outputs.logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"])

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss.detach().cpu())
        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {
                key: torch.stack(val).mean() for key, val in self.scorer.losses.items()
            }
            self.log_dict({f"eval/loss_{key}": val for key, val in logging_loss.items()})
            self.log("train/acc", self.scorer.acc)
            self.scorer.reset()
        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"].float())

        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"])
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    @overrides
    def _build_model(self):
        """"""
        config = AutoConfig.from_pretrained(
            self.pretrained_model, num_labels=self.model_config.num_labels
        )
        model = AutoAdapterModel.from_pretrained(self.pretrained_model, config=config)

        model.add_adapter(self.task_name, config=self.adapter_config_name)
        model.add_classification_head(
            head_name=self.task_name,
            num_labels=self.model_config.num_labels,
            id2label={i: label for i, label in enumerate(self.labels)},
        )
        model.set_active_adapters([self.task_name])
        model.train_adapter([self.task_name])
        self.model = model
