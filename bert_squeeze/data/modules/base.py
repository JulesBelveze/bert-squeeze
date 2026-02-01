import logging
from typing import Optional, Union

import datasets
import lightning.pytorch as pl


class BaseDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for loading datasets.

    This class is responsible for preparing and loading datasets to be used in PyTorch Lightning
    training and validation steps. It inherits from `pl.LightningDataModule`.

    Attributes:
        dataset_config (Dict): A configuration dictionary with keys such as 'path' and 'percent'.
        dataset (datasets.DatasetDict): The dataset loaded from `datasets.load_dataset` method,
            potentially subsetted based on 'percent' in `dataset_config`.
    """

    def prepare_data(self) -> None:
        """
        Loads and potentially subsets the dataset as specified by `dataset_config`.

        If 'percent' key is present in `dataset_config`, subsets each split in the dataset
        to the specified percentage. Prints a log message upon successful loading of the dataset.

        Returns:
            None
        """
        dataset = datasets.load_dataset(self.dataset_config.path, trust_remote_code=True)
        dataset = self._normalize_splits(dataset)

        if "percent" in self.dataset_config:
            dataset = self._subset_percent(dataset, self.dataset_config.percent)

        dataset = self._ensure_required_splits(dataset)

        self.dataset = dataset
        logging.info(f"DatasetDict '{self.dataset_config.path}' successfully loaded.")

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

    def _normalize_splits(
        self, dataset: Union[datasets.DatasetDict, datasets.Dataset]
    ) -> datasets.DatasetDict:
        if isinstance(dataset, datasets.Dataset):
            dataset = datasets.DatasetDict({"train": dataset})

        if "val" in dataset:
            if "validation" in dataset:
                logging.warning(
                    "Both 'val' and 'validation' splits found; dropping 'val' to avoid "
                    "ambiguity."
                )
                dataset.pop("val", None)
            else:
                dataset["validation"] = dataset.pop("val")
        return dataset

    def _ensure_required_splits(
        self, dataset: datasets.DatasetDict
    ) -> datasets.DatasetDict:
        if "train" not in dataset:
            raise ValueError("Dataset must include a 'train' split.")

        has_validation = "validation" in dataset
        has_test = "test" in dataset

        if has_validation and has_test:
            return dataset

        val_size = self._coerce_fraction(self.dataset_config.get("val_size"), 0.1)
        test_size = self._coerce_fraction(self.dataset_config.get("test_size"), 0.1)
        seed = int(self.dataset_config.get("seed", 42))
        stratify_by = self.dataset_config.get("stratify_by_column")

        train_dataset = dataset["train"]

        if not has_validation and not has_test:
            split = self._split_train_dataset(
                train_dataset,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                stratify_by_column=stratify_by,
            )
            dataset["train"] = split["train"]
            if "validation" in split:
                dataset["validation"] = split["validation"]
            if "test" in split:
                dataset["test"] = split["test"]
            return dataset

        if not has_validation:
            split = self._split_train_dataset(
                train_dataset,
                val_size=val_size,
                test_size=0.0,
                seed=seed,
                stratify_by_column=stratify_by,
            )
            dataset["train"] = split["train"]
            dataset["validation"] = split["validation"]
            return dataset

        split = self._split_train_dataset(
            train_dataset,
            val_size=0.0,
            test_size=test_size,
            seed=seed,
            stratify_by_column=stratify_by,
        )
        dataset["train"] = split["train"]
        dataset["test"] = split["test"]
        return dataset

    @staticmethod
    def _coerce_fraction(value: Optional[float], default: float) -> float:
        if value is None:
            return default
        fraction = float(value)
        if fraction <= 0:
            return default
        if fraction >= 1:
            raise ValueError("Split fractions must be in (0, 1).")
        return fraction

    @staticmethod
    def _split_train_dataset(
        dataset: datasets.Dataset,
        *,
        val_size: float,
        test_size: float,
        seed: int,
        stratify_by_column: Optional[str],
    ) -> datasets.DatasetDict:
        total = val_size + test_size
        if total <= 0:
            return datasets.DatasetDict({"train": dataset})
        if total >= 1:
            raise ValueError("val_size + test_size must be in (0, 1).")

        first_split = dataset.train_test_split(
            test_size=total,
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
        train_dataset = first_split["train"]
        remainder = first_split["test"]

        if val_size <= 0:
            return datasets.DatasetDict({"train": train_dataset, "test": remainder})
        if test_size <= 0:
            return datasets.DatasetDict({"train": train_dataset, "validation": remainder})

        val_ratio = val_size / total
        val_test_split = remainder.train_test_split(
            train_size=val_ratio,
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
        return datasets.DatasetDict(
            {
                "train": train_dataset,
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            }
        )
