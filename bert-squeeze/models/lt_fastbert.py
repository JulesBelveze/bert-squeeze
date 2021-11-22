import logging
import os
from collections import defaultdict
from typing import Dict, Tuple, Union, List

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from overrides import overrides
from transformers.models.bert.modeling_bert import BertEmbeddings, BertConfig
from transformers import AutoModel

from .base_lt_module import BaseModule
from .custom_transformers.fastbert import FastBertGraph
from ..utils.types import FastBertLoss


class LtFastBert(BaseModule):
    def __init__(self, training_config: DictConfig, pretrained_model: str, num_labels: int, scorer_type: str, **kwargs):
        super().__init__(training_config, num_labels, pretrained_model, scorer_type, **kwargs)
        self.training_stage = getattr(kwargs, "training_stage", 0)

        self._build_model()
        if self.training_stage == 0:
            self._load_pretrained_bert_model(getattr(kwargs, "pretrained_model_path", None))

    @overrides
    def _build_model(self):
        self.embeddings = BertEmbeddings(self.model_config)
        self.encoder = FastBertGraph(self.model_config)

    def _load_pretrained_bert_model(self, pretrained_model_path: str = None) -> None:
        """"""
        if pretrained_model_path is None:
            tmp_model = AutoModel.from_pretrained(self.pretrained_model)
            tmp_model.save_pretrained("tmp_model")

            pretrained_model_path = os.path.join("tmp_model", "pytorch_model.bin")

        pretrained_model_weights = torch.load(pretrained_model_path, map_location='cpu')
        self.load_state_dict(pretrained_model_weights, strict=False)

    @overrides
    def prune_heads(self, heads_to_prune):
        raise NotImplementedError()

    @overrides
    def freeze_encoder(self):
        """Freeze backbone and final classifier"""
        for name, p in self.named_parameters():
            if "branch_classifier" not in name:
                p.requires_grad = False
        logging.info("Backbone and final classification layer successfully froze.")

    @overrides
    def unfreeze_encoder(self):
        """Unfreeze backbone and final classifier"""
        for name, p in self.named_parameters():
            if "branch_classifier" not in name:
                p.requires_grad = True
        logging.info("Backbone and final classification layer successfully unfroze.")

    @overrides
    def loss(self, outputs: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """"""
        kl_divergences = defaultdict(float)
        # only compute the loss for the final classifier
        if self.training_stage == 0:
            loss = self.objective(outputs, labels)

        # compute the KL divergence for every single student.
        else:
            loss = 0.0
            teacher_logits = outputs[-1]
            teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
            for i, student_logits in enumerate(outputs[:-1]):
                student_prob = F.softmax(student_logits, dim=-1)
                student_log_prob = F.log_softmax(student_logits, dim=-1)
                kl_div = torch.sum(student_prob * (student_log_prob - teacher_log_prob), 1)
                kl_div = torch.mean(kl_div)
                kl_divergences[f"kl_layer_{i}"] = kl_div.detach().cpu()
                loss += kl_div
        return FastBertLoss(**dict({"full_loss": loss}, **kl_divergences))

    @overrides
    def forward(self, input_ids: torch.Tensor = None, token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None, inference: torch.Tensor = False, labels: torch.Tensor = None,
                inference_speed: float = 0.5, training_stage: int = 0, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            inference=inference,
            inference_speed=inference_speed,
            training_stage=training_stage
        )
        return output

    def shared_step(self, batch: Dict[str, torch.Tensor]) \
            -> Tuple[FastBertLoss, Union[List[torch.Tensor], torch.Tensor]]:
        """Common steps for train/val/test step"""
        inputs = {"input_ids": batch["input_ids"],
                  "attention_mask": batch["attention_mask"],
                  "token_type_ids": batch["token_type_ids"],
                  "training_stage": self.training_stage}

        outputs = self.forward(**inputs)
        losses = self.loss(outputs, batch["labels"])

        if self.training_stage == 0:
            # "outputs" is logits from the last classification layer
            logits = outputs
        else:
            # outputs[-1] is log prob from the last classification layer
            # outputs[:-1] is list of all the student classification layers logits
            logits = outputs[:-1]
        return losses, logits

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs):
        losses, logits = self.shared_step(batch)
        self.scorer.add(logits, batch["labels"], losses)
        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {key: torch.stack(val).mean() for key, val in self.scorer.losses.items()}
            for key, value in logging_loss.items():
                self.logger.experiment[f"train/{key}"].log(value=value, step=self.global_step)

            self.logger.experiment["train/acc"].log(self.scorer.acc, step=self.global_step)
            self.scorer.reset()

        return {"loss": losses.full_loss}

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        losses, logits = self.shared_step(batch)
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), losses)
        return {"loss": losses.full_loss, "logits": logits.cpu()}

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        losses, logits = self.shared_step(batch)
        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), losses)
        return {"loss": losses.full_loss, "logits": logits.cpu()}
