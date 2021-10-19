import logging
from typing import Optional

import datasets
import pytorch_lightning as pl
from omegaconf import DictConfig
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TransformerDataModule(pl.LightningDataModule):
    """DataModule for Transformer-based models."""

    def __init__(self, dataset_config: DictConfig, tokenizer_name: str, max_length: int, **kwargs):
        super().__init__()
        self.dataset_config = dataset_config
        self.text_col = dataset_config.text_col
        self.label_col = dataset_config.label_col

        self.max_length = max_length
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.dataset = None
        self.train = None
        self.test = None
        self.val = None

    def load_dataset(self) -> None:
        """Load dataset"""
        if self.dataset_config.is_local:
            self.dataset = datasets.load_dataset(
                resource_filename("bert-squeeze", f"data/{self.dataset_config.name}_dataset.py"),
                self.dataset_config.split
            )
        else:
            self.dataset = datasets.load_dataset(self.dataset_config.name, self.dataset_config.split)
        logging.info(f"Dataset '{self.dataset_config.name}' successfully loaded.")

    def featurize(self) -> datasets.DatasetDict:
        """Featurize dataset"""
        tokenized_dataset = self.dataset.map(
            lambda x: self.tokenizer(x[self.text_col], padding="max_length", max_length=self.max_length,
                                     truncation=True)
        )
        tokenized_dataset = tokenized_dataset.remove_columns([self.text_col])

        if self.label_col != "labels":
            tokenized_dataset = tokenized_dataset.rename_column(self.label_col, "labels")

        tokenized_dataset.set_format(type='torch', columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
        return tokenized_dataset

    def prepare_data(self) -> None:
        """"""
        self.load_dataset()

    def setup(self, stage: Optional[str] = None):
        featurized_dataset = self.featurize()
        print(featurized_dataset)
        self.train = featurized_dataset["train"]
        self.val = featurized_dataset["validation"]
        self.test = featurized_dataset["test"]

    def _collate_fn(self):
        def _collate(examples):
            return self.tokenizer.pad(examples, return_tensors="pt")

        return _collate

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader"""
        return DataLoader(self.train, collate_fn=self._collate_fn(), batch_size=self.train_batch_size, drop_last=True,
                          num_workers=0)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(self.test, collate_fn=self._collate_fn(), batch_size=self.eval_batch_size, drop_last=True,
                          num_workers=0)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        return DataLoader(self.val, collate_fn=self._collate_fn(), batch_size=self.eval_batch_size, drop_last=True,
                          num_workers=0)
