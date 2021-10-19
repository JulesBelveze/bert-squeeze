from typing import List

import torch
import torch.nn.functional as F
from overrides import overrides
from omegaconf import DictConfig

from .base_lt_module import BaseModule


class LtCustomLabse(BaseModule):
    def __init__(self, training_config: DictConfig, model_config: str, num_labels: int, **kwargs):
        super().__init__(training_config, model_config, num_labels, **kwargs)

    @overrides
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        """
        :param input_ids: sentence or sentences represented as tokens
        :param attention_mask: tells the model which tokens in the input_ids are words and which are padding.
                               1 indicates a token and 0 indicates padding.
        :param token_type_ids: used when there are two sentences that need to be part of the input. It indicate which
                               tokens are part of sentence1 and which are part of sentence2.
        :return:

        For specifications about model output, please refer to:
        https://github.com/huggingface/transformers/blob/b01f451ca38695c60175b34d245997ef4d18231d/src/transformers/modeling_outputs.py#L153
        """
        model_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        logits = self.classifier(embeddings)
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, logits = self.shared_step(batch)

        self.scorer.add(logits, batch["labels"], loss.detach().cpu())
        if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"loss_{key}"].log(value)

            self.logger.experiment["train/loss"].log(value=logging_loss, step=self.global_step)
            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()

        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    def shared_step(self, batch):
        inputs = {"input_ids": batch["input_ids"],
                  "attention_mask": batch["attention_mask"],
                  "token_type_ids": None}

        logits = self.forward(**inputs)
        loss = self.loss(logits, batch["labels"])
        return loss, logits

    def training_epoch_end(self, _):
        pass

    def validation_epoch_end(self, test_step_outputs: List[dict]):
        all_logits = torch.cat([pred["logits"] for pred in test_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()

    def test_epoch_end(self, outputs) -> None:
        print(self.test_scorer.get_table())
        self.test_scorer.reset()
