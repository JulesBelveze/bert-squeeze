from typing import Optional

import datasets
from omegaconf import DictConfig
from overrides import overrides
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from .base import BaseDataModule


class TransformerDataModule(BaseDataModule):
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

    def setup(self, stage: Optional[str] = None):
        """"""
        featurized_dataset = self.featurize()
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
        dataset_config.label_col = None
        self.translation_col = dataset_config.get("translation_col", "translation")
        super().__init__(dataset_config, tokenizer_name, max_length, **kwargs)

    @overrides
    def featurize(self) -> datasets.DatasetDict:
        """
        Returns:
            DatasetDict: featurized dataset
        """
        self.dataset = self.dataset.filter(lambda x: x[self.translation_col] is not None)
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
                    x[self.translation_col],
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                ).items()
            }
        )
        tokenized_dataset = tokenized_dataset.remove_columns(
            [self.text_col, self.translation_col]
        )

        columns = [
            "input_ids",
            "attention_mask",
            "translation_input_ids",
            "translation_attention_mask",
        ]
        if "distilbert" not in self.tokenizer.name_or_path:
            columns += ["token_type_ids", "translation_token_type_ids"]

        tokenized_dataset.set_format(type='torch', columns=columns)
        return tokenized_dataset


class Seq2SeqTransformerDataModule(BaseDataModule):
    """
    DataModule for Transformer-based models on sequence-to-sequence tasks.

    Args:
        dataset_config (HydraConfig):
            dataset configuration
        tokenizer_name (str):
            name of the pre-trained tokenizer to use
        max_target_length (int):
            maximum sequence length of the targeted text
       max_source_length (int):
            maximum sequence length of the source text
    """

    def __init__(
        self,
        dataset_config: DictConfig,
        tokenizer_name: str,
        max_target_length: int,
        max_source_length: int,
        **kwargs,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.source_col = dataset_config.source_col
        self.target_col = dataset_config.target_col

        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.dataset = None
        self.train = None
        self.test = None
        self.val = None

    def featurize(self) -> datasets.DatasetDict:
        """
        Returns:
            DatasetDict: featurized dataset
        """
        tokenized_dataset = self.dataset.map(
            lambda x: self.tokenizer(
                x[self.source_col],
                padding=False,
                max_length=self.max_source_length,
                truncation=True,
            )
        )
        with self.tokenizer.as_target_tokenizer():
            tokenized_dataset = tokenized_dataset.map(
                lambda x: {
                    "labels": self.tokenizer(
                        x[self.target_col],
                        padding=False,
                        max_length=self.max_target_length,
                        truncation=True,
                    )["input_ids"]
                }
            )
        columns = ["input_ids", "attention_mask", "labels"]
        if not any(
            [
                model_name in self.tokenizer.name_or_path
                for model_name in ["distilbert", "t5"]
            ]
        ):
            columns += ["token_type_ids"]

        columns_to_keep = [self.target_col, self.source_col] + columns
        for split, split_dataset in tokenized_dataset.items():
            columns_to_del = set(split_dataset.column_names) - set(columns_to_keep)
            tokenized_dataset[split] = split_dataset.remove_columns(list(columns_to_del))
        tokenized_dataset.set_format(type='torch', columns=columns)
        return tokenized_dataset

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
            collate_fn=self._collate_fn(),
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
            collate_fn=self._collate_fn(),
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
            collate_fn=self._collate_fn(),
            batch_size=self.eval_batch_size,
            drop_last=True,
        )

    def _collate_fn(self):
        """Helper function to merge a list of samples into a batch of Tensors"""

        def _collate(examples):
            return self.tokenizer.pad(examples, return_tensors="pt")

        return _collate
