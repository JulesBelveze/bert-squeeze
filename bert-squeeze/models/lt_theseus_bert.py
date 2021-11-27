from typing import Union

import torch
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoConfig

from .base_lt_module import BaseModule
from .custom_transformers import TheseusBertModel
from ..utils.schedulers.theseus_schedulers import ConstantReplacementScheduler, LinearReplacementScheduler


class LtTheseusBert(BaseModule):
    def __init__(self, training_config: DictConfig, pretrained_model: str, num_labels: int,
                 replacement_scheduler: DictConfig, **kwargs):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)

        self._build_model()
        self.replacement_scheduler = {
            "linear": LinearReplacementScheduler,
            "constant": ConstantReplacementScheduler
        }[replacement_scheduler.type](self.encoder.encoder,
                                      **{k: v for k, v in replacement_scheduler.items() if k != "type"})

    @overrides
    def _build_model(self):
        encoder = TheseusBertModel(AutoConfig.from_pretrained(self.pretrained_model))
        encoder.from_pretrained(self.pretrained_model)
        encoder.encoder.init_successor_layers()
        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels)
        )

    @overrides
    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None, position_ids: torch.Tensor = None, head_mask: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None, **kwargs):
        """
        :param input_ids: sentence or sentences represented as tokens
        :param attention_mask: tells the model which tokens in the input_ids are words and which are padding.
                               1 indicates a token and 0 indicates padding.
        :param token_type_ids: used when there are two sentences that need to be part of the input. It indicate which
                               tokens are part of sentence1 and which are part of sentence2.
        :param position_ids: indices of positions of each input sequence tokens in the position embeddings. Selected
                             in the range ``[0, config.max_position_embeddings - 1]
        :param head_mask: mask to nullify selected heads of the self-attention modules
        :param inputs_embeds:
        :param output_attentions:
        :return:

        For specifications about model output, please refer to:
        https://github.com/huggingface/transformers/blob/b01f451ca38695c60175b34d245997ef4d18231d/src/transformers/modeling_outputs.py#L153
        """
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        self.replacement_scheduler.step()
        loss, logits = self.shared_step(batch)

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss.detach().cpu())
        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"train/loss_{key}"].log(value=value, step=self.global_step)

            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()

        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}

    def shared_step(self, batch):
        inputs = {"input_ids": batch["input_ids"],
                  "attention_mask": batch["attention_mask"],
                  "token_type_ids": batch["token_type_ids"]}

        logits = self.forward(**inputs)
        loss = self.loss(logits, batch["labels"])
        return loss, logits
