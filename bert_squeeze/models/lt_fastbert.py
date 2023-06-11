import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from overrides import overrides
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings

from ..utils.types import FastBertLoss
from .base_lt_module import BaseTransformerModule
from .custom_transformers.fastbert import FastBertGraph


class LtFastBert(BaseTransformerModule):
    """
    Lightning module to fine-tune a FastBert based model on a sequence classification
    task (see `models.custom_transformers.fastbert.py`) for detailed explanation.

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels
        pretrained_model (str):
            name of the pretrained Transformer model to use
    """

    def __init__(
        self,
        training_config: DictConfig,
        num_labels: int,
        pretrained_model: str,
        **kwargs,
    ):
        super().__init__(training_config, num_labels, pretrained_model, **kwargs)
        self.training_stage = getattr(kwargs, "training_stage", 0)

        self._build_model()
        if self.training_stage == 0:
            self._load_pretrained_bert_model(
                getattr(kwargs, "pretrained_model_path", None)
            )

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inference: torch.Tensor = False,
        inference_speed: float = 0.5,
        training_stage: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor):
                sentence or sentences represented as tokens
            attention_mask (torch.Tensor):
                tells the model which tokens in the input_ids are words and which are padding.
                1 indicates a token and 0 indicates padding.
            token_type_ids (torch.Tensor):
                used when there are two sentences that need to be part of the input. It indicates which
                tokens are part of sentence1 and which are part of sentence2.
            inference (bool):
                whether the model is inference mode. If yes, after each student layer we compute
                the uncertainty and feeds it to the next layer until enough information is
                has been collected.
            inference_speed (float):
                threshold to distinguish high and low uncertainty. The higher the `inference_speed`
                the faster and less accurate the model is.
            training_stage (training_stage):
                current training stage. If equals to 0, fine-tuning both the major backbone and
                the teacher classifier, while the student layers are frozen. If equals to 1 the
                backbone is frozen and only the student layers are fine-tuned.
        Returns:
            torch.Tensor: logits obtained from the model pass
        """
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
            embeddings=embedding_output,
            attention_mask=extended_attention_mask,
            device=self.device,
            inference=inference,
            inference_speed=inference_speed,
            training_stage=training_stage,
        )
        return output

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "training_stage": self.training_stage,
        }

        outputs = self.forward(**inputs)
        losses = self.loss(outputs, batch["labels"])

        if self.training_stage == 0:
            # "outputs" is logits from the last classification layer
            logits = outputs
        else:
            # outputs[-1] is log prob from the last classification layer
            # outputs[:-1] is list of all the student classification layers logits
            logits = outputs[:-1]

        self.scorer.add(logits, batch["labels"], losses)
        if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
            logging_loss = {
                key: torch.stack(val).mean() for key, val in self.scorer.losses.items()
            }
            self.log_dict({f"train/loss_{key}": val for key, val in logging_loss.items()})
            self.log("train/acc", self.scorer.acc)
            self.scorer.reset()

        return losses.full_loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> None:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "training_stage": self.training_stage,
        }

        outputs = self.forward(**inputs)
        losses = self.loss(outputs, batch["labels"])

        if self.training_stage == 0:
            # "outputs" is logits from the last classification layer
            logits = outputs
        else:
            # outputs[-1] is log prob from the last classification layer
            # outputs[:-1] is list of all the student classification layers logits
            logits = outputs[:-1]
        self.valid_scorer.add(logits.cpu(), batch["labels"].cpu(), losses)
        self.validation_step_outputs.append(
            {"loss": losses.full_loss, "logits": logits.cpu()}
        )

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> None:
        """"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "training_stage": self.training_stage,
        }

        outputs = self.forward(**inputs)
        losses = self.loss(outputs, batch["labels"])

        # outputs[-1] is log prob from the last classification layer
        # outputs[:-1] is list of all the student classification layers logits
        logits = outputs[:-1]

        self.test_scorer.add(logits.cpu(), batch["labels"].cpu(), losses)
        self.test_step_outputs.append({"loss": losses.full_loss, "logits": logits.cpu()})

    @overrides
    def _build_model(self):
        """"""
        self.embeddings = BertEmbeddings(self.model_config)
        self.encoder = FastBertGraph(self.model_config)

    def _load_pretrained_bert_model(self, pretrained_model_path: str = None) -> None:
        """
        Loads the pretrained weights into the model.

        Args:
            pretrained_model_path (str):
                If given downloads a model from the HF hub and saves the weights before
                loading them into the model. If `None` it directly loads the model from
                a local path.
        """
        if pretrained_model_path is None:
            tmp_model = AutoModel.from_pretrained(self.pretrained_model)
            tmp_model.save_pretrained("tmp_model")

            pretrained_model_path = os.path.join("tmp_model", "pytorch_model.bin")

        pretrained_model_weights = torch.load(pretrained_model_path, map_location='cpu')
        self.load_state_dict(pretrained_model_weights, strict=False)

    @overrides
    def prune_heads(self, heads_to_prune):
        """"""
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
    def loss(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        labels: torch.Tensor,
        *args,
        **kwargs,
    ) -> FastBertLoss:
        """
        Handles the loss computation part.

        When `training_stage=0` we simply optimize the backbone model using `self.objective`.
        When `training_stage=1` we compute the Kullback-Leibler divergence between the distribution
        of every student and the backbone model.

        Args:
            outputs (Union[torch.Tensor, List[torch.Tensor]]):
                When `training_stage=0` the predicted logits of the backbone model. When
                `training_stage=1` the list of predicted logits by the student layers.
            labels (torch.Tensor):
                ground truth labels
        Returns:
            FastBertLoss: object containing the different layers loss
        """
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
                kl_div = torch.sum(
                    student_prob * (student_log_prob - teacher_log_prob), 1
                )
                kl_div = torch.mean(kl_div)
                kl_divergences[f"kl_layer_{i}"] = kl_div.detach().cpu()
                loss += kl_div
        return FastBertLoss(**dict({"full_loss": loss}, **kl_divergences))
