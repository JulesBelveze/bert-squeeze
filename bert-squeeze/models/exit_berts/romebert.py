from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from overrides import overrides
from torch.nn import CrossEntropyLoss

import numpy as np
from ...utils.types import KDLossOutput
from ...utils.losses import KDLoss
from .layers import RomeBertModel
from ..base_lt_module import BaseModule
from ...utils.errors import RampException


class RomeBert(BaseModule):

    def __init__(self, config, model_config):
        super().__init__(config, model_config)
        self.automatic_optimization = False
        self.gradient_projection = config.train.gradient_projection

    @overrides
    def _build_model(self):
        self.bert = RomeBertModel(self.model_config)
        self.num_layers = len(self.bert.encoder.layer)
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model_config.hidden_size, self.model_config.num_labels)

        self.bert.init_weights()
        self.bert.encoder.set_early_exit_entropy(self.config.train.early_exit_entropy)
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
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.params.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    @overrides
    def loss(self, logits: torch.Tensor, labels: torch.Tensor, ramps_exits: torch.Tensor, exit_layer: int,
             train_ramps: bool = False, output_layer: int = -1, gamma=0.9, temper=3.0, *args, **kwargs) \
            -> Tuple[KDLossOutput, Optional[torch.Tensor]]:
        ramps_entropy, ramps_logits_all = [], []
        for ramps_exit in ramps_exits:
            ramps_logits_all.append(ramps_exit[0])
            ramps_entropy.append(ramps_exit[2])

        loss_fct = KDLoss(len(ramps_exits), gamma, temper, self.model_config.num_labels)
        soft_labels = logits.detach()
        loss = loss_fct(
            outputs=logits,
            highway_outputs=ramps_logits_all,
            targets=labels,
            soft_targets=soft_labels
        )
        return loss

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        opt = self.optimizers()
        opt.zero_grad()

        loss, logits = self.shared_step(batch)

        last_loss = loss.last_loss
        multi_loss = loss.multi_loss
        full_loss = loss.full_loss
        if torch.cuda.device_count() > 1:
            last_loss = last_loss.mean()
            multi_loss = multi_loss.mean()
            full_loss = full_loss.mean()

        if self.gradient_projection:
            self.manual_backward(last_loss, retain_graph=True)

            var_list_last, grad_list_last, grad_size_last = [], [], []
            for name, param in self.named_parameters():
                if 'encoder.layer' in name and 'layer.11' not in name and 'LayerNorm' not in name and 'bias' not in name and param.grad is not None:
                    var_list_last.append(name)
                    grad_list_last.append(param.grad.view(-1))
                    grad_size_last.append(param.grad.shape)
                    '''optional: setting param gradient to zero'''
                    # param.grad.data.zero_()
            grad_list_last = torch.cat(grad_list_last)

            self.manual_backward(multi_loss)

            var_list_multi, grad_list_multi, grad_size_multi = [], [], []
            for name, param in self.named_parameters():
                if 'encoder.layer' in name and 'layer.11' not in name and 'LayerNorm' not in name and 'bias' not in name and param.grad is not None:
                    var_list_multi.append(name)
                    grad_list_multi.append(param.grad.view(-1))
                    grad_size_multi.append(param.grad.shape)
            grad_list_multi = torch.cat(grad_list_multi)
            assert var_list_last == var_list_multi

            grad_list_multi = grad_list_multi - grad_list_last

            inner_product = torch.sum(grad_list_multi * grad_list_last)
            proj_direction = inner_product / torch.sum(grad_list_last * grad_list_last)
            grad_list_multi = grad_list_multi - torch.min(proj_direction, torch.zeros([1])) * grad_list_last

            # Unpack flattened projected gradients back to their original shapes.
            start_idx = 0
            idx = 0
            for name, param in self.named_parameters():
                if name in var_list_multi:
                    grad_shape = grad_size_multi[idx]
                    flatten_dim = np.prod([grad_shape[i] for i in range(len(grad_shape))])
                    proj_grad_last = torch.reshape(grad_list_last[start_idx:start_idx + flatten_dim], grad_shape)
                    proj_grad_multi = torch.reshape(grad_list_multi[start_idx:start_idx + flatten_dim], grad_shape)
                    param.grad = proj_grad_last.detach() + proj_grad_multi.detach()
                    start_idx += flatten_dim
                    idx += 1
        else:
            self.manual_backward(full_loss)

        opt.step()

        self.scorer.add(logits.detach().cpu(), batch["labels"], loss=loss)
        if self.config.general.logging_steps > 0 and self.global_step % self.config.general.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"loss_{key}"].log(value)
            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()

        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), loss=loss)
        return {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        loss, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), loss=loss)
        return {"loss": loss, "logits": logits.cpu(), "labels": batch["labels"].cpu()}

    def shared_step(self, batch):
        inputs = {"input_ids": batch["input_ids"],
                  "attention_mask": batch["attention_mask"],
                  "token_type_ids": batch["token_type_ids"]}

        logits, ramps_exits, exit_layer = self.forward(**inputs)
        loss = self.loss(logits=logits, labels=batch["labels"], exit_layer=exit_layer, ramps_exits=ramps_exits)
        return loss, logits

    def validation_epoch_end(self, outputs: List[dict]):
        all_logits = torch.cat([pred["logits"] for pred in outputs])
        all_probs = F.softmax(all_logits, dim=-1)
        labels_probs = all_probs.numpy()

        epoch_loss = {key: torch.stack(val).mean() for key, val in self.valid_scorer.losses.items()}
        self.log_eval_report(labels_probs)
        self.valid_scorer.reset()
        return epoch_loss

    def test_epoch_end(self, _) -> None:
        print(self.test_scorer.get_table())
        self.test_scorer.reset()
