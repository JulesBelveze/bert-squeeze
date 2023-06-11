import logging
from typing import Optional

import datasets
import lightning.pytorch as pl
from omegaconf import DictConfig
from overrides import overrides
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TransformerDataModule(pl.LightningDataModule):
    """
    DataModule for Transformer-based models.

    Args:
        dataset_config (HydraConfig):
            dataset configuration
        tokenizer_name (str):
            name of the pre-trained tokenizer to use
        max_length (int):
            maximum sequence length of the inputs to the tokenizer
    """

    def __init__(
        self, dataset_config: DictConfig, tokenizer_name: str, max_length: int, **kwargs
    ):
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

    def featurize(self) -> datasets.DatasetDict:
        """
        Returns:
            DatasetDict: featurized dataset
        """
        tokenized_dataset = self.dataset.map(
            lambda x: self.tokenizer(
                x[self.text_col],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        )
        tokenized_dataset = tokenized_dataset.remove_columns([self.text_col])

        if self.label_col != "labels":
            tokenized_dataset = tokenized_dataset.rename_column(self.label_col, "labels")

        columns = ["input_ids", "attention_mask", "labels"]
        if "distilbert" not in self.tokenizer.name_or_path:
            columns += ["token_type_ids"]

        tokenized_dataset.set_format(type='torch', columns=columns)
        return tokenized_dataset

    def prepare_data(self) -> None:
        """"""
        self.load_dataset()

    def setup(self, stage: Optional[str] = None):
        """"""
        featurized_dataset = self.featurize()
        print(featurized_dataset)
        self.train = featurized_dataset["train"]
        self.val = featurized_dataset["validation"]
        self.test = featurized_dataset["test"]

    def _collate_fn(self):
        """Helper function to merge a list of samples into a batch of Tensors"""

        def _collate(examples):
            return self.tokenizer.pad(examples, return_tensors="pt")

        return _collate

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(
            self.train,
            # collate_fn=self._collate_fn(),
            batch_size=self.train_batch_size,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: test dataloader
        """
        return DataLoader(
            self.test,
            # collate_fn=self._collate_fn(),
            batch_size=self.eval_batch_size,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val,
            # collate_fn=self._collate_fn(),
            batch_size=self.eval_batch_size,
            drop_last=True,
        )


class TransformerParallelDataModule(TransformerDataModule):
    """
    DataModule for parallel dataset for Transformer-based models.

    Args:
        dataset_config (HydraConfig):
            dataset configuration
        tokenizer_name (str):
            name of the pre-trained tokenizer to use
        max_length (int):
            maximum sequence length of the inputs to the tokenizer
    """

    def __init__(
        self, dataset_config: DictConfig, tokenizer_name: str, max_length: int, **kwargs
    ):
        dataset_config.text_col = "text"
        dataset_config.label_col = None
        super().__init__(dataset_config, tokenizer_name, max_length, **kwargs)

    @overrides
    def featurize(self) -> datasets.DatasetDict:
        """
        Returns:
            DatasetDict: featurized dataset
        """
        tokenized_dataset = self.dataset.map(
            lambda x: self.tokenizer(
                x[self.text_col],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        )
        tokenized_dataset = tokenized_dataset.map(
            lambda x: {
                "translation_" + name: value
                for name, value in self.tokenizer(
                    x["translation"],
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                ).items()
            }
        )
        tokenized_dataset = tokenized_dataset.remove_columns(
            [self.text_col, "translation"]
        )

        columns = [
            "input_ids",
            "attention_mask",
            "translation_input_ids",
            "translation_input_ids",
        ]
        if "distilbert" not in self.tokenizer.name_or_path:
            columns += ["token_type_ids", "translation_token_type_ids"]

        tokenized_dataset.set_format(type='torch', columns=columns)
        return tokenized_dataset
