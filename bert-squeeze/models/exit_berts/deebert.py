from transformers import AutoConfig
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from overrides import overrides
from torch.nn import CrossEntropyLoss

from .deebert_layers import DeeBertModel
from ..base_lt_module import BaseModule
from ...utils.errors import RampException
from omegaconf import DictConfig


class DeeBert(BaseModule):
    def __init__(self, training_config: DictConfig, pretrained_model: str, num_labels: int, **kwargs):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)
        self.train_highway = training_config.train_highway
        self._build_model()

    @overrides
    def _build_model(self):
        self.bert = DeeBertModel(self.model_config)
        self.num_layers = len(self.bert.encoder.layer)
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model_config.hidden_size, self.model_config.num_labels)

        self.bert.init_weights()
        self.bert.encoder.set_early_exit_entropy(self.config.early_exit_entropy)
        self.bert.init_highway_pooler()

    @overrides
    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None, position_ids: torch.Tensor = None, head_mask: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None, labels: torch.Tensor = None, output_layer: int = -1,
                train_highway: bool = False, **kwargs):
        try:
            exit_layer = self.num_layers
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds
            )  # sequence_output, pooled_output, (hidden_states), (attentions), ramps exits
            pooled_output = outputs.pooled_output

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            ramps_exits = outputs.ramps_exits

        except RampException as e:
            # In case it exited earlier
            # Note: this doesn't happen during fine-tuning
            outputs = e.message
            ramps_exits = outputs[-1]
            exit_layer = e.exit_layer
            logits = outputs[0]

        return logits, ramps_exits, exit_layer

    @overrides
    def _get_optimizer_parameters(self):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']

        if self.config.discriminative_learning:
            if isinstance(self.config.learning_rates, ListConfig) and len(self.config.learning_rates) > 1:
                groups = [(f'layer.{i}.', self.config.learning_rates[i]) for i in range(12)]
            else:
                lr = self.config.learning_rates[0] if isinstance(self.config.learning_rates,
                                                                 ListConfig) else self.config.learning_rates
                groups = [(f'layer.{i}.', lr * pow(self.config.layer_lr_decay, 11 - i)) for i in range(12)]

            group_all = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters, decay_optimizer_parameters = [], []
            for g, l in groups:
                no_decay_optimizer_parameters.append(
                    {'params': [p for n, p in self.named_parameters() if ("highway" not in n) and
                                not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                     'weight_decay_rate': self.config.weight_decay, 'lr': l}
                )
                decay_optimizer_parameters.append(
                    {'params': [p for n, p in self.named_parameters() if ("highway" not in n) and
                                any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                     'weight_decay_rate': 0.0, 'lr': l}
                )

            group_all_parameters = [
                {'params': [p for n, p in self.named_parameters() if ("highway" not in n) and
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
                 'weight_decay_rate': self.config.weight_decay},
                {'params': [p for n, p in self.named_parameters() if ("highway" not in n) and
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
                 'weight_decay_rate': 0.0},
            ]
            optimizer_grouped_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters \
                                           + group_all_parameters
        else:
            if self.config.train_highway:
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in self.named_parameters() if
                                ("highway" in n) and (not any(nd in n for nd in no_decay))],
                     'weight_decay': self.config.weight_decay},
                    {'params': [p for n, p in self.named_parameters() if
                                ("highway" in n) and (any(nd in n for nd in no_decay))],
                     'weight_decay': 0.0}
                ]
            else:
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in self.named_parameters() if
                                ("highway" not in n) and (not any(nd in n for nd in no_decay))],
                     'weight_decay': self.config.weight_decay},
                    {'params': [p for n, p in self.named_parameters() if
                                ("highway" not in n) and (any(nd in n for nd in no_decay))],
                     'weight_decay': 0.0}
                ]
        return optimizer_grouped_parameters

    @overrides
    def loss(self, logits: torch.Tensor, labels: torch.Tensor, ramps_exits: torch.Tensor, exit_layer: int,
             train_ramps: bool = False, output_layer: int = -1, *args, **kwargs):
        if train_ramps:
            # We want to fine-tune each individual ramp
            ramps_entropy, ramps_logits_all = [], []
            ramps_losses = []
            for ramps_exit in ramps_exits:
                ramps_logits = ramps_exit[0]
                if not self.training:
                    ramps_logits_all.append(ramps_logits)
                    ramps_entropy.append(ramps_exit[2])

                loss_fct = CrossEntropyLoss()
                ramps_loss = loss_fct(ramps_logits.view(-1, self.model_config.num_labels),
                                      labels.view(-1))
                ramps_losses.append(ramps_loss)
            # We train all but the last off ramp (corresponds to stage 2 in paper)
            loss = sum(ramps_losses[:-1])
        else:
            # We only train the last off ramp
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model_config.num_labels), labels.view(-1))
        return loss

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss, logits = self.shared_step(batch)

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss.detach().cpu())
        if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"train/loss_{key}"].log(value)

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
                  "token_type_ids": batch["token_type_ids"],
                  "train_highway": self.train_highway}
        logits, ramps_exits, exit_layer = self.forward(**inputs)
        return self.loss(
            logits=logits,
            labels=batch["labels"],
            exit_layer=exit_layer,
            ramps_exits=ramps_exits
        ), logits

    def validation_epoch_end(self, outputs: List[dict]):
        all_logits = torch.cat([pred["logits"] for pred in outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()
        return torch.stack([out["loss"] for out in outputs]).mean()

    def test_epoch_end(self, _) -> None:
        print(self.test_scorer.get_table())
        self.test_scorer.reset()
