import logging
from typing import Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from overrides import overrides
from torch.nn import CrossEntropyLoss

from bert_squeeze.utils.scorers import Scorer

from .base_lt_module import BaseSequenceClassificationTransformerModule
from .custom_transformers.berxit import BerxitModel


class LtBerxit(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune a BERxiT-style model for sequence classification.

    This mirrors LtDeeBert's integration, exposing the same training/inference
    behavior and configuration hooks (e.g., train_highway, early_exit_entropy).
    """

    def __init__(
        self,
        training_config: DictConfig,
        pretrained_model: str,
        num_labels: int,
        model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        scorer: Scorer = None,
        **kwargs,
    ):
        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )
        self.train_highway = training_config.train_highway
        self.train_gates = getattr(training_config, "train_gates", False)
        self._build_model()

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], int, Optional[Tuple[torch.Tensor]]]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        if self.training:
            exit_layer = self.num_layers
            pooled_output = outputs.pooled_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            ramps_exits = outputs.ramps_exits
            gates_logits = outputs.gates_logits
        else:
            ramps_exits = outputs.ramps_exits
            exit_layer = outputs.exit_layer
            logits = outputs.logits
            gates_logits = None

        return logits, ramps_exits, exit_layer, gates_logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits, ramps_exits, _, gates_logits = self.forward(**inputs)
        loss = self.loss(
            logits=logits,
            labels=batch["labels"],
            train_ramps=self.train_highway,
            ramps_exits=ramps_exits,
            train_gates=self.train_gates,
            gates_logits=gates_logits,
        )

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss.detach().cpu())
        if (
            self.config.logging_steps > 0
            and self.global_step % self.config.logging_steps == 0
        ):
            logging_loss = {
                key: torch.stack(val).mean() for key, val in self.scorer.losses.items()
            }
            self.log_dict({f"train/loss_{key}": val for key, val in logging_loss.items()})
            self.log("train/acc", self.scorer.acc)
            self.scorer.reset()

        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits, ramps_exits, _, gates_logits = self.forward(**inputs)
        loss = self.loss(
            logits=logits,
            labels=batch["labels"],
            ramps_exits=ramps_exits,
            train_ramps=self.train_highway,
            train_gates=self.train_gates,
            gates_logits=gates_logits,
        )
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.validation_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        all_logits = torch.cat([pred["logits"] for pred in self.validation_step_outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits, _, _ = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"], train_ramps=self.train_highway)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss.cpu())
        self.test_step_outputs.append(
            {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def predict_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        self.bert.set_inference_mode(inference=True)
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        logits, _, _ = self.forward(**inputs)
        preds = torch.softmax(logits, dim=-1)
        return preds

    @overrides
    def _get_optimizer_parameters(self) -> List[Dict]:
        # Mirror LtDeeBert grouping to keep behavior consistent
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']

        if self.config.discriminative_learning:
            if (
                isinstance(self.config.learning_rates, ListConfig)
                and len(self.config.learning_rates) > 1
            ):
                groups = [(f'layer.{i}.', self.config.learning_rates[i]) for i in range(12)]
            else:
                lr = (
                    self.config.learning_rates[0]
                    if isinstance(self.config.learning_rates, ListConfig)
                    else self.config.learning_rates
                )
                groups = [
                    (f'layer.{i}.', lr * pow(self.config.layer_lr_decay, 11 - i))
                    for i in range(12)
                ]

            group_all = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters, decay_optimizer_parameters = [], []
            for g, l in groups:
                no_decay_optimizer_parameters.append(
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if ("highway" not in n)
                            and not any(nd in n for nd in no_decay)
                            and any(nd in n for nd in [g])
                        ],
                        'weight_decay_rate': self.config.weight_decay,
                        'lr': l,
                    }
                )
                decay_optimizer_parameters.append(
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if ("highway" not in n)
                            and any(nd in n for nd in no_decay)
                            and any(nd in n for nd in [g])
                        ],
                        'weight_decay_rate': 0.0,
                        'lr': l,
                    }
                )

            group_all_parameters = [
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
                        if ("highway" not in n)
                        and not any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay_rate': self.config.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
                        if ("highway" not in n)
                        and any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay_rate': 0.0,
                },
            ]
            optimizer_grouped_parameters = (
                no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters
            )
        else:
            if self.config.train_highway:
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if ("highway" in n) and (not any(nd in n for nd in no_decay))
                        ],
                        'weight_decay': self.config.weight_decay,
                    },
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if ("highway" in n) and (any(nd in n for nd in no_decay))
                        ],
                        'weight_decay': 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if ("highway" not in n)
                            and (not any(nd in n for nd in no_decay))
                        ],
                        'weight_decay': self.config.weight_decay,
                    },
                    {
                        'params': [
                            p
                            for n, p in self.named_parameters()
                            if ("highway" not in n) and (any(nd in n for nd in no_decay))
                        ],
                        'weight_decay': 0.0,
                    },
                ]
        return optimizer_grouped_parameters

    @overrides
    def loss(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor = None,
        ramps_exits: Tuple[torch.Tensor] = None,
        train_ramps: bool = False,
        train_gates: bool = False,
        gates_logits: Optional[Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Same ramp loss mechanics as LtDeeBert for consistency
        if train_ramps:
            ramps_losses = []
            for ramps_exit in ramps_exits[:-1]:
                ramps_logits = ramps_exit.logits
                loss_fct = CrossEntropyLoss()
                ramps_loss = loss_fct(
                    ramps_logits.view(-1, self.model_config.num_labels), labels.view(-1)
                )
                ramps_losses.append(ramps_loss)
            loss = sum(ramps_losses)
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model_config.num_labels), labels.view(-1)
            )
        # Optional: add gate loss using pseudo-labels from final ramp
        if train_gates and gates_logits is not None:
            with torch.no_grad():
                final_logits = ramps_exits[-1].logits  # [B, C]
                final_pred = final_logits.argmax(dim=-1)  # [B]
            bce = torch.nn.BCEWithLogitsLoss()
            gate_losses = []
            for i, gate_logit in enumerate(gates_logits[:-1]):
                layer_pred = ramps_exits[i].logits.argmax(dim=-1)  # [B]
                target = (layer_pred == final_pred).float().unsqueeze(-1)  # [B,1]
                gate_losses.append(bce(gate_logit, target))
            if gate_losses:
                loss = loss + sum(gate_losses)
        return loss

    def _build_model(self):
        # Pass BERxiT-specific hyperparams via HF config attributes
        if not hasattr(self.model_config, "gate_hidden_dim"):
            self.model_config.gate_hidden_dim = getattr(self.config, "gate_hidden_dim", 32)
        self.bert = BerxitModel(self.model_config)
        self.num_layers = len(self.bert.encoder.layer)
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels),
        )

        self.bert.init_weights()
        self.bert.encoder.set_early_exit_entropy(self.config.early_exit_entropy)
        # Optional: set gate thresholds for early exit
        if hasattr(self.config, "gate_thresholds"):
            self.bert.set_exit_gate_thresholds(self.config.gate_thresholds)
        self.bert.init_highway_pooler()
