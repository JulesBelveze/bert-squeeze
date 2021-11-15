import torch
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoModel

from .base_lt_module import BaseModule


class LtCustomDistilBert(BaseModule):
    def __init__(self, training_config: DictConfig, pretrained_model: str, num_labels: int, **kwargs):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)
        self._build_model()

    @overrides
    def _build_model(self):
        self.encoder = AutoModel.from_pretrained(self.pretrained_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.seq_classif_dropout),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels)
        )

    @overrides
    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None,
                output_attentions: bool = False, **kwargs):
        """"""
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions
        )
        hidden_state = outputs[0]
        logits = self.classifier(hidden_state)

        if output_attentions:
            return logits, outputs.attentions
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
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
                  "attention_mask": batch["attention_mask"]}

        logits = self.forward(**inputs)
        loss = self.loss(logits, batch["labels"])
        return loss, logits
