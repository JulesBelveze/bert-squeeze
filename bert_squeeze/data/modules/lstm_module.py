import logging
import re
from typing import Any, Dict, List, Optional

import datasets
import lightning.pytorch as pl
import numpy as np
import torch
from datasets import DatasetDict
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader

from ...utils.vocabulary import Vocabulary


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Helper function to merge a list of samples into a batch of Tensors

    Args:
        batch (List[Dict[str, Any]]):
            List of samples to merge to form a batch
    Returns:
        Dict[str, torch.Tensor]: batch of tensor features
    """
    features = [elt["features"] for elt in batch]
    labels = [elt["labels"] for elt in batch]

    longest = max([len(x) for x in features])
    stacked = np.stack([np.pad(x, (0, longest - len(x))) for x in features])
    return {"features": torch.LongTensor(stacked), "labels": torch.LongTensor(labels)}


class LSTMDataModule(pl.LightningDataModule):
    """
    DataModule for LSTM models

    Args:
        dataset_config (HydraConfig):
            dataset configuration
        max_features (int):
            Number of maximum words to use to build a vocabulary.
    """

    def __init__(
        self,
        dataset_config: HydraConfig,
        max_features: int,
        token_pattern: str = r"(?u)\b\w\w+\b",
        **kwargs,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.text_col = dataset_config.text_col
        self.label_col = dataset_config.label_col

        self.vocabulary = Vocabulary(
            path_to_voc=kwargs.get("path_to_voc"), max_words=max_features
        )
        token_pattern = re.compile(token_pattern)
        self.tokenizer = token_pattern.findall

        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)

        self.dataset = None
        self.train = None
        self.test = None
        self.val = None

    def load_dataset(self) -> None:
        """"""
        if self.dataset_config.is_local:
            self.dataset = datasets.load_dataset(
                self.dataset_config.path, self.dataset_config.split
            )
        else:
            self.dataset = datasets.load_dataset(
                self.dataset_config.path, self.dataset_config.split
            )
        logging.info(f"Dataset '{self.dataset_config.path}' successfully loaded.")

    def clean_str(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Utility method to properly format the `self.text_col` of an example
        and avoid OOV tokens.

        Args:
            example (Dict[str, Any]):
                sample for which to format the `self.text_col` column
        Returns:
            Dict[str, Any]: example with re-formatted text
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", example[self.text_col])
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        example[self.text_col] = string.strip().lower()
        return example

    def tokens_to_ids(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Utility method to tokenize the `self.text_col`. It will add a "features" key
        to `example` which contains a List[int] corresponding to the vocabulary ids
        of each token.

        Args:
            example (Dict[str, Any]):
                sample for which to tokenize `self.text_col`
        Returns:
            Dict[str, Any]: featurized example
        """
        tokenized_doc = self.tokenizer(example[self.text_col])
        ids = [self.vocabulary[token] for token in tokenized_doc]

        example["features"] = ids
        return example

    def featurize(self) -> DatasetDict:
        """
        Returns:
            DatasetDict: featurized dataset
        """
        dataset = self.dataset.map(lambda x: self.clean_str(x), batched=False)

        corpus = self.dataset["train"][self.text_col]
        tokenized_corpus = map(self.tokenizer, corpus)
        self.vocabulary.build_vocabulary(tokenized_corpus)

        featurized_dataset = dataset.map(
            lambda x: self.tokens_to_ids(x), batched=False, remove_columns=[self.text_col]
        )

        if self.label_col != "labels":
            featurized_dataset = featurized_dataset.rename_column(
                self.label_col, "labels"
            )

        featurized_dataset.set_format(type='np', columns=["features", "labels"])
        logging.info("Data successfully featurized and Dataset created.")
        return featurized_dataset

    def prepare_data(self) -> None:
        """"""
        self.load_dataset()

    def setup(self, stage: Optional[str] = None):
        """"""
        featurized_dataset = self.featurize()

        self.train = featurized_dataset["train"]
        self.val = featurized_dataset["validation"]
        self.test = featurized_dataset["test"]

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(
            self.train,
            collate_fn=collate_fn,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: test dataloader
        """
        return DataLoader(
            self.test,
            collate_fn=collate_fn,
            batch_size=self.eval_batch_size,
            drop_last=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val,
            collate_fn=collate_fn,
            batch_size=self.eval_batch_size,
            drop_last=True,
            num_workers=0,
        )
