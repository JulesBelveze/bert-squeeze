from pathlib import Path
from typing import List, Optional, Sequence

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

        raw_paths = dataset_config.get("data_path", None)
        self._uses_data_path = raw_paths is not None
        if raw_paths is None:
            raw_paths = dataset_config.get("path")
        if raw_paths is None:
            raise ValueError("dataset_config must specify 'path' or 'data_path'.")
        if isinstance(raw_paths, str):
            self.data_paths: List[str] = [raw_paths]
        else:
            self.data_paths = list(raw_paths)
        if not self.data_paths:
            raise ValueError("dataset_config.data_path must contain at least one path.")

        data_format = dataset_config.get("data_format")
        if isinstance(data_format, Sequence) and not isinstance(data_format, str):
            self.data_formats = list(data_format)
        elif data_format is None:
            self.data_formats = None
        else:
            self.data_formats = [data_format] * len(self.data_paths)

        self.combine_strategy = dataset_config.get("combine_strategy", "concatenate")

        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.dataset = None
        self.train = None
        self.test = None
        self.val = None

    def prepare_data(self) -> None:
        if (
            not self._uses_data_path
            and len(self.data_paths) == 1
            and self.data_formats is None
        ):
            super().prepare_data()
            return

        datasets_list = []
        formats = self._resolve_formats()
        for path, fmt in zip(self.data_paths, formats):
            datasets_list.append(self._load_dataset(path, fmt))

        combined = (
            datasets_list[0]
            if len(datasets_list) == 1
            else self._combine_datasets(datasets_list)
        )

        if "percent" in self.dataset_config:
            combined = self._subset_percent(combined, self.dataset_config.percent)

        self.dataset = combined

    def _resolve_formats(self) -> List[str]:
        if self.data_formats is not None:
            if len(self.data_formats) != len(self.data_paths):
                raise ValueError(
                    "Length of data_format list must match number of data paths."
                )
            return self.data_formats
        return [self._detect_format(path) for path in self.data_paths]

    def _load_dataset(self, path: str, data_format: str) -> datasets.DatasetDict:
        if data_format == "disk":
            loaded = datasets.load_from_disk(path)
            if isinstance(loaded, datasets.Dataset):
                return datasets.DatasetDict({"train": loaded})
            return loaded
        if data_format == "hub":
            return datasets.load_dataset(path, trust_remote_code=True)
        if data_format in {"json", "jsonl"}:
            return datasets.load_dataset("json", data_files=path)
        if data_format == "csv":
            return datasets.load_dataset("csv", data_files=path)
        raise ValueError(f"Unsupported data_format '{data_format}'.")

    def _combine_datasets(
        self, datasets_list: Sequence[datasets.DatasetDict]
    ) -> datasets.DatasetDict:
        splits = datasets_list[0].keys()
        for ds in datasets_list[1:]:
            if ds.keys() != splits:
                raise ValueError("All datasets must share identical splits.")

        if self.combine_strategy == "concatenate":
            return datasets.DatasetDict(
                {
                    split: datasets.concatenate_datasets(
                        [ds[split] for ds in datasets_list]
                    )
                    for split in splits
                }
            )
        if self.combine_strategy == "interleave":
            return datasets.DatasetDict(
                {
                    split: datasets.interleave_datasets(
                        [ds[split] for ds in datasets_list]
                    )
                    for split in splits
                }
            )
        raise ValueError(
            f"Unknown combine_strategy '{self.combine_strategy}'. "
            "Expected 'concatenate' or 'interleave'."
        )

    @staticmethod
    def _subset_percent(
        dataset: datasets.DatasetDict, percent: float
    ) -> datasets.DatasetDict:
        if percent <= 0 or percent > 100:
            raise ValueError("percent must be in (0, 100].")
        return datasets.DatasetDict(
            {
                split: split_dataset.select(
                    range(int(len(split_dataset) * percent / 100))
                )
                for split, split_dataset in dataset.items()
            }
        )

    @staticmethod
    def _detect_format(path: str) -> str:
        candidate = Path(path)
        if candidate.is_dir():
            return "disk"
        if candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix in {".jsonl", ".json"}:
                return "json"
            if suffix == ".csv":
                return "csv"
            raise ValueError(
                f"Unable to infer data_format from file extension '{suffix}' for '{path}'. "
                "Set dataset_config.data_format explicitly."
            )
        return "hub"

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
