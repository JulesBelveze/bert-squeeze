# This code is heavily inspired by: https://github.com/BitVoyage/FastBERT
# The main difference relies on the fact that I'm trying to use HuggingFace's
# 'transformers' components as much as possible.

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import BertLayer, BertSelfAttention


class FastBertClassifier(nn.Module):
    """
    This module is a "branch", also defined as student classifier in the paper,
    that mimics the teacher's architecture. A `FastBertClassifier` layer is
    squeezed between each Transformer block to enable early outputs in simple cases.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a model according to the specified
            arguments, defining the model architecture.
    """

    def __init__(self, config: PretrainedConfig):
        super(FastBertClassifier, self).__init__()

        cls_hidden_size = config.hidden_size  # might need to be reduced
        num_class = config.num_labels

        self.dense_narrow = nn.Linear(config.hidden_size, cls_hidden_size)
        self.selfAttention = BertSelfAttention(config)
        self.dense_prelogits = nn.Linear(cls_hidden_size, cls_hidden_size)
        self.dense_logits = nn.Linear(cls_hidden_size, num_class)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """"""
        states_output = self.dense_narrow(hidden_states)
        states_output = self.selfAttention(states_output, attention_mask)
        token_cls_output = states_output[0][:, 0]
        prelogits = self.dense_prelogits(token_cls_output)
        logits = self.dense_logits(prelogits)
        return logits


class FastBertGraph(nn.Module):
    """
    Implementation of the FastBERT model presented by in "FastBERT: a Self-distilling
    BERT with Adaptive Inference Time" https://arxiv.org/abs/2004.02178.

    FastBERT enables the user to adapt the inference speed by avoiding redundant
    calculations. This is achieved by squeezing student classifier within every
    Transformer block of the backbone.

    The training procedure is done in two stage. First the backbone as well as the final
    classifier are fine-tuned while the student classifiers are kept frozen. In the
    second stage the model's knowledge is self-distilled to the student classifiers
    which predictions are compared to the soft-labeled teacher outputs.

    Args:
        config (PretrainedConfig):
            Configuration to use to instantiate a BERT model according to the
            specified arguments, defining the model architecture.
    """

    def __init__(self, config: PretrainedConfig):
        super(FastBertGraph, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_classifier = FastBertClassifier(config)

        # creating branches
        self.layer_classifiers = nn.ModuleDict()
        for i in range(config.num_hidden_layers - 1):
            self.layer_classifiers['branch_classifier_' + str(i)] = FastBertClassifier(
                config
            )

        # creating backbone classifier
        self.layer_classifiers['final_classifier'] = self.layer_classifier

        self.ce_loss_fct = nn.CrossEntropyLoss()
        self.num_class = torch.tensor(config.num_labels, dtype=torch.float32)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        device: str,
        inference: bool = False,
        inference_speed: float = 0.5,
        training_stage: int = 1,
    ) -> Union[Tuple[torch.Tensor, int], List[torch.Tensor]]:
        """
        Args:
            embeddings (torch.Tensor):
                embeddings constructed from word, position and token_type.
            attention_mask (torch.Tensor):
                tells the model which tokens in the input_ids are words and which are padding.
                1 indicates a token and 0 indicates padding.
            device (str):
                device to move the tensors to.
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
            Union[Tuple[torch.Tensor, int], List[torch.Tensor]]: at inference time the model returns
                the predicted logits along with the index of the exit layer. At training time the
                model returns a list containing the predicted logits of all the exit layers.
        """
        hidden_states = (embeddings,)
        # In the Inference stage, if the uncertainty of the i-th student is low it will be
        # returned earlier.
        # To handle batches we have to compute the uncertainty and regroup the low-confidence
        # examples into a new batch that is fed to the next layer.
        if inference:
            # positions will keep track of the original position of each element in the
            # batch when elements will be removed
            final_probs = torch.zeros((hidden_states[0].shape[0], 2), device=device)
            positions = torch.arange(
                start=0, end=hidden_states[0].shape[0], device=device
            ).long()

            for i, (layer_module, (k, layer_classifier_module)) in enumerate(
                zip(self.layer, self.layer_classifiers.items())
            ):
                hidden_states = layer_module(hidden_states[0], attention_mask)
                logits = layer_classifier_module(hidden_states[0])
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                uncertain = torch.sum(prob * log_prob, 1) / (-torch.log(self.num_class))

                # checking if there's enough information
                enough_info = uncertain < inference_speed

                right_pos = positions[enough_info]
                final_probs[right_pos] = prob[enough_info]

                hidden_states = (hidden_states[0][~enough_info],)
                attention_mask = attention_mask[~enough_info]

                # if we have processed all the samples
                if hidden_states[0].shape[0] == 0:
                    return final_probs, i

                positions = positions[
                    ~enough_info
                ]  # updating the positions to fit the new batch

            return final_probs, i

        # Training phase: the first phase corresponds to the backbone training
        # the second phase to the branches training (actual distillation)
        else:
            # Initial training (consistent with normal fine-tuning)
            if training_stage == 0:
                for layer_module in self.layer:
                    hidden_states = layer_module(hidden_states[0], attention_mask)

                logits = self.layer_classifier(hidden_states[0])
                return logits

            # Distillation training, the KL divergence of students and teachers in each layer is used as loss
            else:
                all_encoder_layers = []
                for layer_module in self.layer:
                    hidden_states = layer_module(hidden_states[0], attention_mask)
                    all_encoder_layers.append(hidden_states[0])

                all_logits = []
                for encoder_layer, (k, layer_classifier_module) in zip(
                    all_encoder_layers, self.layer_classifiers.items()
                ):
                    layer_logits = layer_classifier_module(encoder_layer)
                    all_logits.append(layer_logits)

                return all_logits
