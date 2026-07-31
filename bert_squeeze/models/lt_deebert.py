from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from overrides import overrides
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig

from bert_squeeze.utils.scorers import Scorer
from bert_squeeze.utils.types import RampOutput, SequenceClassificationOutput

from .base_lt_module import BaseSequenceClassificationTransformerModule
from .custom_transformers.deebert import DeeBertModel


class LtDeeBert(BaseSequenceClassificationTransformerModule):
    """
    Lightning module to fine-tune a DeeBert based model on a sequence classification
    task (see `models.custom_transformers.deebert.py`) for detailed explanation.

    Args:
        training_config (DictConfig):
            training configuration
        num_labels (int):
            number of labels
        pretrained_model (str):
            name of the pretrained Transformer model to use
        model (Optional[Union[pl.LightningModule, nn.Module]]):
            optional instantiated model
        scorer (Scorer):
            helper object to compute performance metrics during training
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
        if model is None:
            model = DeeBertModel.from_pretrained(
                pretrained_model,
                config=AutoConfig.from_pretrained(
                    pretrained_model, num_labels=num_labels
                ),
            )

        super().__init__(
            training_config, pretrained_model, num_labels, model, scorer, **kwargs
        )
        self.train_highway = training_config.train_highway
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
    ) -> Tuple[torch.Tensor, Sequence[RampOutput], int]:
        """
        During training, we pass the hidden states through all layers and store all the off-ramps
        outputs as well as the final classification layer.
        During inference, we try to pass the hidden states through the whole BertLayer and OffRamps stack
        which is exited as soon as the entropy of one layer is lower than a given threshold.

        Args:
            input_ids (torch.Tensor):
                sentence or sentences represented as tokens
            attention_mask (torch.Tensor):
                tells the model which tokens in the input_ids are words and which are padding.
                1 indicates a token and 0 indicates padding.
            token_type_ids (torch.Tensor):
                used when there are two sentences that need to be part of the input. It indicates which
                tokens are part of sentence1 and which are part of sentence2.
            position_ids (torch.Tensor):
                indices of positions of each input sequence tokens in the position embeddings. Selected
                in the range ``[0, config.max_position_embeddings - 1]
            head_mask (torch.Tensor):
                mask to nullify selected heads of the self-attention modules
        Returns:
            torch.Tensor:
                output of the classification layer which uses the last ramp output during training and
                the output of the exited ramp during inference.
            Tuple[torch.Tensor]:
                iterable containing all ramp exits
            int:
                index of the exited ramp
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        if not self.bert.encoder.inference:
            exit_layer = self.num_layers
            pooled_output = outputs.pooled_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            ramps_exits = outputs.ramps_exits
        else:
            ramps_exits = outputs.ramps_exits
            exit_layer = outputs.exit_layer
            logits = outputs.logits

        return logits, ramps_exits, exit_layer

    @overrides
    def _classification_output(
        self, batch: Dict[str, torch.Tensor]
    ) -> SequenceClassificationOutput:
        logits, ramps_exits, exit_layer = self.forward(**self._model_inputs(batch))
        return SequenceClassificationOutput(
            logits=logits,
            ramps_exits=ramps_exits,
            exit_layer=exit_layer,
        )

    @overrides
    def _classification_loss(
        self,
        output: SequenceClassificationOutput,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(
            logits=output.logits,
            labels=labels,
            train_ramps=self.train_highway,
            ramps_exits=output.ramps_exits,
        )

    @overrides
    def _before_prediction_step(self) -> None:
        self.bert.set_inference_mode(inference=True)

    @overrides
    def _get_optimizer_parameters(self) -> List[Dict]:
        """
        Method that defines the parameter to optimize.

        Returns:
            List[Dict]: group of parameters to optimize
        """
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight', 'layer_norm.weight']

        if self.config.discriminative_learning:
            if (
                isinstance(self.config.learning_rates, ListConfig)
                and len(self.config.learning_rates) > 1
            ):
                groups = [
                    (f'layer.{i}.', self.config.learning_rates[i]) for i in range(12)
                ]
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
                        'weight_decay': self.config.weight_decay,
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
                        'weight_decay': 0.0,
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
                    'weight_decay': self.config.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in self.named_parameters()
                        if ("highway" not in n)
                        and any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in group_all)
                    ],
                    'weight_decay': 0.0,
                },
            ]
            optimizer_grouped_parameters = (
                no_decay_optimizer_parameters
                + decay_optimizer_parameters
                + group_all_parameters
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
        logits: Optional[torch.Tensor] = None,
        ramps_exits: Optional[Sequence[RampOutput]] = None,
        train_ramps: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Handles the loss computation part.

        If `train_ramps=False` we only use the logits of the final classification layer to compute
        the cross entropy. If `train_ramps=True` we add up all the cross entropies of the off-ramps.

        Args:
            labels (torch.Tensor):
                ground truth labels
            ramps_exits (Tuple[torch.Tensor]):
                list containing the predicted logits from all the off-ramps
            logits (torch.Tensor):
                predicted logits by the final classification layer
            train_ramps (bool):
                whether to train the off-ramps or the final classification layer.
        Returns:

        """
        # We want to fine-tune each individual ramp
        if train_ramps:
            if ramps_exits is None or len(ramps_exits) < 2:
                raise ValueError("Ramp training requires at least two ramp outputs.")
            ramps_losses: List[torch.Tensor] = []
            # We train all but the last off-ramp (corresponds to stage 2 in paper)
            for ramps_exit in ramps_exits[:-1]:
                ramps_logits = ramps_exit.logits

                loss_fct = CrossEntropyLoss()
                ramps_loss = loss_fct(
                    ramps_logits.view(-1, self.model_config.num_labels), labels.view(-1)
                )
                ramps_losses.append(ramps_loss)

            loss = torch.stack(ramps_losses).sum()
        else:
            if logits is None:
                raise ValueError(
                    "Classifier logits are required when ramps are disabled."
                )
            # We only train the last off-ramp
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model_config.num_labels), labels.view(-1)
            )
        return loss

    def _build_model(self):
        """"""
        self.bert = self.model
        self.num_layers = len(self.bert.encoder.layer)
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.model_config.hidden_dropout_prob),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.model_config.hidden_size),
            torch.nn.Linear(self.model_config.hidden_size, self.model_config.num_labels),
        )

        self.bert.encoder.set_early_exit_entropy(self.config.early_exit_entropy)
        self.bert.init_highway_pooler()
