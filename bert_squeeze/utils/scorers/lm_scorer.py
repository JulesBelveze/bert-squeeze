import copy
from collections import defaultdict
from typing import Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from tabulate import tabulate
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from bert_squeeze.utils.types import DistillationLoss

MAX_CLIP_VALUE = 1e8
IGNORE_INDEX = -100


def _get_fallback_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for token_id_attr in ("pad_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, token_id_attr, None)
        if token_id is not None:
            return int(token_id)
    return 0


def _replace_ignore_index(
    token_ids: torch.Tensor, *, replacement_token_id: Optional[int]
) -> torch.Tensor:
    if replacement_token_id is None:
        return token_ids
    if not (token_ids == IGNORE_INDEX).any().item():
        return token_ids

    token_ids = token_ids.clone()
    token_ids[token_ids == IGNORE_INDEX] = replacement_token_id
    return token_ids


class LMScorer(object):
    """
    Scorer for language modeling tasks
    """

    def __init__(self, tokenizer_name: str = None, do_mismatch: bool = True):
        """"""
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        self.do_mismatch = do_mismatch

        self.losses = defaultdict(list)
        self.metrics = defaultdict(list)
        self.mismatches = []

    @property
    def perplexity(self):
        """"""
        if not self.metrics["perplexity"]:
            return float("nan")
        return torch.stack(self.metrics["perplexity"]).mean().item()

    @staticmethod
    def postprocess_text(*args: List[str]):
        """"""
        return tuple([item.strip() for item in lst] for lst in args)

    def add(
        self,
        loss: Union[torch.Tensor, DistillationLoss] = None,
        predicted_tokens: torch.Tensor = None,
        labels: torch.Tensor = None,
        input_ids: torch.Tensor = None,
    ):
        """"""
        with torch.no_grad():
            if loss is None:
                return

            loss_tensor = (
                loss.full_loss.detach()
                if isinstance(loss, DistillationLoss)
                else loss.detach()
            )
            self.losses["global"].append(loss_tensor.cpu())

            perplexity = torch.exp(loss_tensor).clamp(max=MAX_CLIP_VALUE)
            self.metrics["perplexity"].append(perplexity.cpu())

            if not self.do_mismatch:
                return
            if self.tokenizer is None:
                return
            if predicted_tokens is None or labels is None or input_ids is None:
                return

            decoded_preds = self.tokenizer.batch_decode(
                predicted_tokens, skip_special_tokens=True
            )

            labels_for_decode = _replace_ignore_index(
                labels, replacement_token_id=self.tokenizer.pad_token_id
            )

            decoded_labels = self.tokenizer.batch_decode(
                labels_for_decode, skip_special_tokens=True
            )
            input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            for pred, label, text in zip(decoded_preds, decoded_labels, input_texts):
                if pred != label:
                    self.mismatches.append(
                        {"prediction": pred, "truth": label, "text": text}
                    )

    def result(self):
        """"""
        return {"perplexity": self.perplexity}

    def reset(self):
        """"""
        self.losses = defaultdict(list)
        self.metrics = defaultdict(list)
        self.mismatches = []

    def to_dict(self) -> Dict[str, float]:
        """
        Returns all the accessible metrics within a dict where the key is the metric name
        and the value is the metric.

        Returns:
            Dict[str, float]: dict of metrics
        """
        return self.result()

    def get_table(self) -> str:
        """
        Method to format all the metrics into a pretty table.

        Returns:
            str: prettyfied table summarizing all the metrics
        """
        table = [[key, value] for key, value in self.to_dict().items()]
        return tabulate(
            table,
            headers=["metrics", "value"],
            tablefmt="fancy_grid",
        )


class SummarizationScorer(object):
    """
    Scorer for summarization tasks
    """

    def __init__(self, tokenizer_name: str = None, do_mismatch: bool = True):
        """"""
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        self.do_mismatch = do_mismatch

        self.losses = defaultdict(list)
        self.mismatches = []
        self.metrics = evaluate.load("rouge")

    def __deepcopy__(self, memo=None):
        # `metrics` is a whole module that can't be deep-copied
        cls = self.__class__
        result = cls.__new__(cls)
        memo = memo or {}
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != 'metrics':
                setattr(result, k, copy.deepcopy(v, memo))

        result.metrics = evaluate.load("rouge")
        return result

    @staticmethod
    def postprocess_text(*args: List[str]):
        """"""
        return tuple([item.strip() for item in lst] for lst in args)

    def add(
        self,
        loss: Union[torch.Tensor, DistillationLoss] = None,
        predicted_tokens: torch.Tensor = None,
        labels: torch.Tensor = None,
        input_ids: torch.Tensor = None,
    ):
        """
        Updates the score with the new loss and the desired metrics.

        Args:
            loss (torch.Tensor):
                optimization loss
            labels (torch.Tensor):
                ground truth labels
            predicted_tokens (torch.Tensor):
                predicted tokens
            input_ids (torch.Tensor):
                token ids of the input sentences
        """
        with torch.no_grad():
            if isinstance(loss, torch.Tensor):
                self.losses["global"].append(loss.cpu())
            else:
                self.losses["global"].append(loss.full_loss.cpu())

            if self.do_mismatch and predicted_tokens is not None:
                decoded_preds = self.tokenizer.batch_decode(
                    predicted_tokens, skip_special_tokens=True
                )

                labels_for_decode = labels
                if labels_for_decode is not None:
                    replacement_token_id = None
                    if (labels_for_decode == IGNORE_INDEX).any():
                        replacement_token_id = _get_fallback_pad_token_id(self.tokenizer)
                    labels_for_decode = _replace_ignore_index(
                        labels_for_decode, replacement_token_id=replacement_token_id
                    )

                decoded_labels = self.tokenizer.batch_decode(
                    labels_for_decode, skip_special_tokens=True
                )
                input_ids_cpu = input_ids.cpu()
                input_ids = np.where(
                    input_ids_cpu != IGNORE_INDEX,
                    input_ids_cpu,
                    self.tokenizer.pad_token_id,
                )
                input_texts = self.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )

                decoded_preds, decoded_labels, input_texts = self.postprocess_text(
                    decoded_preds, decoded_labels, input_texts
                )
                self.metrics.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )

                for pred, label, text in zip(
                    predicted_tokens, labels_for_decode, input_ids
                ):
                    predicted_kw = self.tokenizer.decode(pred, skip_special_tokens=True)
                    truth = self.tokenizer.decode(label, skip_special_tokens=True)
                    initial_text = self.tokenizer.decode(text, skip_special_tokens=True)

                    if predicted_kw != truth:
                        self.mismatches.append(
                            {
                                "prediction": predicted_kw,
                                "truth": truth,
                                "text": initial_text,
                            }
                        )

    def result(self):
        """"""
        result = self.metrics.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

    def reset(self):
        """"""
        self.losses = defaultdict(list)
        self.mismatches = []

    def to_dict(self) -> Dict[str, float]:
        """
        Returns all the accessible metrics within a dict where the key is the metric name
        and the value is the metric.

        Returns:
            Dict[str, float]: dict of metrics
        """
        return self.result()

    @staticmethod
    def get_table(results: Dict[str, float]) -> str:
        """
        Method to format all the metrics into a pretty table.

        Args:
            results (Dict[str, float]): dictionary of metrics
        Returns:
            str: prettyfied table summarizing all the metrics
        """
        table = []
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, torch.Tensor):
                value = value.item()
            table.append([key, value])
        return tabulate(
            table,
            headers=["metrics", "value"],
            tablefmt="fancy_grid",
        )
