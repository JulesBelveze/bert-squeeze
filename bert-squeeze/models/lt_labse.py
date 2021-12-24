import torch
from omegaconf import DictConfig
from overrides import overrides

from .base_lt_module import BaseModule
from .custom_transformers import CustomLabseModel


class LtCustomLabse(BaseModule):
    def __init__(self, training_config: DictConfig, pretrained_model: str, num_labels: int, **kwargs):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)
        self._build_model()

    @overrides
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, head_mask=None,
                output_attentions: bool = False, **kwargs):
        """
        :param input_ids: sentence or sentences represented as tokens
        :param attention_mask: tells the model which tokens in the input_ids are words and which are padding.
                               1 indicates a token and 0 indicates padding.
        :param token_type_ids: used when there are two sentences that need to be part of the input. It indicate which
                               tokens are part of sentence1 and which are part of sentence2.
        :param head_mask:
        :param output_attentions:
        :return:

        For specifications about model output, please refer to:
        https://github.com/huggingface/transformers/blob/b01f451ca38695c60175b34d245997ef4d18231d/src/transformers/modeling_outputs.py#L153
        """
        model_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions
        )

        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        logits = self.classifier(embeddings)
        if output_attentions:
            return logits, model_output.attentions
        return logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, logits = self.shared_step(batch)

        self.scorer.add(logits, batch["labels"], loss.detach().cpu())
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
        return {"loss": loss, "logits": logits.cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        return {"loss": loss, "logits": logits.cpu()}

    @overrides
    def _build_model(self):
        self.encoder = CustomLabseModel.from_pretrained(self.pretrained_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels)
        )

    def shared_step(self, batch):
        inputs = {"input_ids": batch["input_ids"],
                  "attention_mask": batch["attention_mask"],
                  "token_type_ids": batch["token_type_ids"]}

        logits = self.forward(**inputs)
        loss = self.loss(logits, batch["labels"])
        return loss, logits
