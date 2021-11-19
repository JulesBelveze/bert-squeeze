# This code is heavily inspired by: https://github.com/BitVoyage/FastBERT
# The main difference relies on the fact that I'm trying to use HuggingFace's
# 'transformers' components as much as possible.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertLayer, BertSelfAttention, BertConfig


class FastBertClassifier(nn.Module):
    def __init__(self, config: BertConfig):
        super(FastBertClassifier, self).__init__()

        cls_hidden_size = config.hidden_size  # might need to be reduced
        num_class = config.num_labels

        self.dense_narrow = nn.Linear(config.hidden_size, cls_hidden_size)
        self.selfAttention = BertSelfAttention(config)
        self.dense_prelogits = nn.Linear(cls_hidden_size, cls_hidden_size)
        self.dense_logits = nn.Linear(cls_hidden_size, num_class)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        states_output = self.dense_narrow(hidden_states)
        states_output = self.selfAttention(states_output, attention_mask)
        token_cls_output = states_output[0][:, 0]
        prelogits = self.dense_prelogits(token_cls_output)
        logits = self.dense_logits(prelogits)
        return logits


class FastBertGraph(nn.Module):
    def __init__(self, bert_config: BertConfig):
        super(FastBertGraph, self).__init__()
        self.layers = nn.ModuleList([BertLayer(bert_config) for _ in range(bert_config.num_hidden_layers)])

        self.layer_classifier = FastBertClassifier(bert_config)
        # creating branches
        self.layer_classifiers = nn.ModuleDict()
        for i in range(bert_config.num_hidden_layers - 1):
            self.layer_classifiers['branch_classifier_' + str(i)] = FastBertClassifier(bert_config)

        # creating backbone classifier
        self.layer_classifiers['final_classifier'] = self.layer_classifier

        self.ce_loss_fct = nn.CrossEntropyLoss()
        self.num_class = torch.tensor(bert_config.num_labels, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, inference: bool = False,
                inference_speed: float = 0.5, training_stage: int = 1):
        """"""
        hidden_states = (hidden_states,)
        # In the Inference stage, if the uncertainty of the i-th student is low
        # it will be returned early.
        if inference:
            uncertain_infos = []
            for i, (layer_module, (k, layer_classifier_module)) in enumerate(
                    zip(self.layers, self.layer_classifiers.items())):
                hidden_states = layer_module(hidden_states[0], attention_mask)
                logits = layer_classifier_module(hidden_states[0])
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                uncertain = torch.sum(prob * log_prob, 1) / (-torch.log(self.num_class))
                uncertain_infos.append([uncertain, prob])

                # return early results
                if uncertain < inference_speed:
                    return prob, i, uncertain_infos
            return prob, i, uncertain_infos

        # Training phase: the first phase corresponds to the backbone training
        # the second phase to the branches training (actual distillation)
        else:
            # Initial training (consistent with normal fine-tuning)
            if training_stage == 0:
                for layer_module in self.layers:
                    hidden_states = layer_module(hidden_states[0], attention_mask)

                logits = self.layer_classifier(hidden_states[0])
                return logits

            # Distillation training, the KL divergence of students and teachers in each layer is used as loss
            else:
                all_encoder_layers = []
                for layer_module in self.layers:
                    hidden_states = layer_module(hidden_states[0], attention_mask)
                    all_encoder_layers.append(hidden_states[0])

                all_logits = []
                for encoder_layer, (k, layer_classifier_module) in \
                        zip(all_encoder_layers, self.layer_classifiers.items()):
                    layer_logits = layer_classifier_module(encoder_layer)
                    all_logits.append(layer_logits)

                return all_logits
