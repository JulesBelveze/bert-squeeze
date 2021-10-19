# import torch
# from overrides import overrides
# from typing import Tuple, List
#
# from .distiller import BaseDistiller
# from ..models import BowLogisticRegression, EmbeddingLogisticRegression
#
#
# class Transformer2LR(BaseDistiller):
#     def __init__(self, config, fine_tuned_teacher, **kwargs):
#         super().__init__(config, fine_tuned_teacher, **kwargs)
#
#     @overrides
#     def _build_student(self) -> None:
#         assert self.config.student.type == "lr"
#         if self.config.student.model == "bow":
#             self.student = BowLogisticRegression(
#                 vocab_size=self.config.student.vocab_size,
#                 num_labels=self.config.student.num_labels
#             )
#         elif self.config.student.model == "embed":
#             self.student = EmbeddingLogisticRegression(
#                 vocab_size=self.config.student.vocab_size,
#                 embed_dim=self.config.student.embed_dim,
#                 num_labels=self.config.student.num_labels
#             )
#         else:
#             raise ValueError(f"Student model '{self.config.student.model}' not supported.")
#
#     def configure_optimizers(self) -> Tuple[List, List]:
#         optimizer = torch.optim.SGD(self.student.parameters(), lr=self.params.learning_rates[0])
#         return [optimizer], []
#
#     @overrides
#     def get_student_logits(self, batch) -> torch.Tensor:
#         student_inputs = {}
#         for key, val in batch.items():
#             if not key.startswith("s_"):
#                 continue
#             if self.config.student.model == "bow" and not isinstance(val, torch.FloatTensor):
#                 val = val.type("torch.FloatTensor")
#             elif self.config.student.model == "embed" and not isinstance(val, torch.LongTensor):
#                 val = val.type("torch.LongTensor")
#             student_inputs[key[2:]] = val
#
#         logits = self.student.forward(**student_inputs)
#         return logits
