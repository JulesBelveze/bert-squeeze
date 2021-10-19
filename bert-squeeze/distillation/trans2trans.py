# import math
# from typing import Tuple, List
# import torch
# from omegaconf import ListConfig
# from overrides import overrides
# from transformers import AutoModelForSequenceClassification, AutoConfig, AdamW, get_linear_schedule_with_warmup
#
# from .distiller import BaseDistiller
# from ..utils.optimizers import BertAdam
#
#
# class Transformer2Transformer(BaseDistiller):
#     def __init__(self, config, fine_tuned_teacher, **kwargs):
#         super().__init__(config, fine_tuned_teacher, **kwargs)
#
#     @overrides
#     def _build_student(self) -> None:
#         assert self.config.student.type == "transformer", \
#             "Student type for 'Transformer2Transformer' must be 'transformer'."
#
#         model_config = AutoConfig.from_pretrained(
#             self.config.student.pretrained_model,
#             num_labels=self.config.student.num_labels,
#             return_dict=False
#         )
#         self.student = AutoModelForSequenceClassification.from_config(model_config)
#
#     @overrides
#     def configure_optimizers(self) -> Tuple[List, List]:
#         def _get_student_parameters():
#             no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
#
#             if self.params.discriminative_learning:
#                 if isinstance(self.params.learning_rates, ListConfig) and len(self.params.learning_rates) > 1:
#                     groups = [(f'layer.{i}.', self.params.learning_rates[i]) for i in range(12)]
#                 else:
#                     lr = self.params.learning_rates[0] if isinstance(self.params.learning_rates,
#                                                                      ListConfig) else self.params.learning_rates
#                     groups = [(f'layer.{i}.', lr * pow(self.params.layer_lr_decay, 11 - i)) for i in range(12)]
#
#                 group_all = [f'layer.{i}.' for i in range(12)]
#                 no_decay_optimizer_parameters, decay_optimizer_parameters = [], []
#                 for g, l in groups:
#                     no_decay_optimizer_parameters.append(
#                         {'params': [p for n, p in self.student.named_parameters() if
#                                     not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
#                          'weight_decay_rate': self.params.weight_decay, 'lr': l}
#                     )
#                     decay_optimizer_parameters.append(
#                         {'params': [p for n, p in self.student.named_parameters() if
#                                     any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
#                          'weight_decay_rate': 0.0, 'lr': l}
#                     )
#
#                 group_all_parameters = [
#                     {'params': [p for n, p in self.student.named_parameters() if
#                                 not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
#                      'weight_decay_rate': self.params.weight_decay},
#                     {'params': [p for n, p in self.student.named_parameters() if
#                                 any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
#                      'weight_decay_rate': 0.0},
#                 ]
#                 optimizer_grouped_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters \
#                                                + group_all_parameters
#             else:
#                 optimizer_grouped_parameters = [
#                     {'params': [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
#                      'weight_decay_rate': self.params.weight_decay},
#                     {'params': [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
#                      'weight_decay_rate': 0.0}
#                 ]
#             return optimizer_grouped_parameters
#
#         optimizer_parameters = _get_student_parameters()
#         if self.params.optimizer == "adamw":
#             optimizer = AdamW(optimizer_parameters, lr=self.params.learning_rates[0],
#                               eps=self.params.adam_eps)
#
#             if self.params.lr_scheduler:
#                 num_training_steps = len(self.train_dataloader()) * self.params.num_epochs // \
#                                      self.params.accumulation_steps
#
#                 warmup_steps = math.ceil(num_training_steps * self.params.warmup_ratio)
#                 scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                             num_warmup_steps=warmup_steps,
#                                                             num_training_steps=num_training_steps)
#                 lr_scheduler = {"scheduler": scheduler, "name": "NeptuneLogger"}
#                 return [optimizer], [lr_scheduler]
#
#         elif self.params.optimizer == "bertadam":
#             num_training_steps = len(self.train_dataloader()) * self.params.num_epochs // \
#                                  self.params.accumulation_steps
#             optimizer = BertAdam(optimizer_parameters, lr=self.params.learning_rates[0],
#                                  warmup=self.params.warmup_ratio, t_total=num_training_steps)
#
#         elif self.params.optimizer == "adam":
#             optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rates[0])
#         else:
#             raise ValueError(f"Optimizer '{self.params.optimizer}' not supported.")
#
#         return [optimizer], []
#
#     @overrides
#     def get_student_logits(self, batch) -> torch.Tensor:
#         student_inputs = {key[2:]: val for key, val in batch.items() if key.startswith("s_")}
#         _, logits = self.student.forward(**student_inputs)
#         return logits
