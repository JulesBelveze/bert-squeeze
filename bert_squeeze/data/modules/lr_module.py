import logging
from typing import Optional

import datasets
import lightning.pytorch as pl
from datasets import Dataset, DatasetDict
from hydra.core.hydra_config import HydraConfig
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader


class LrDataModule(pl.LightningDataModule):
    """
    DataModule for Logistic Regression which can be seen as a bag of n-grams.

    Args:
        dataset_config (HydraConfig):
            dataset configuration
        max_features (int):
            Number of maximum features to use. This will be used to build a vocabulary
            of `max_features` words.
    """

    def __init__(self, dataset_config: HydraConfig, max_features: int, **kwargs):
        super().__init__()
        self.dataset_config = dataset_config
        self.text_col = dataset_config.text_col
        self.label_col = dataset_config.label_col

        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)

        self.cv = CountVectorizer(
            ngram_range=(1, kwargs.get("max_ngrams", 1)), max_features=max_features
        )

        self.dataset = None
        self.train = None
        self.test = None
        self.val = None

    def load_dataset(self) -> None:
        """
        Load dataset
        """
        if self.dataset_config.is_local:
            self.dataset = datasets.load_dataset(
                self.dataset_config.path, self.dataset_config.split
            )
        else:
            self.dataset = datasets.load_dataset(
                self.dataset_config.path, self.dataset_config.split
            )
        logging.info(f"Dataset '{self.dataset_config.path}' successfully loaded.")

    def featurize(self) -> DatasetDict:
        """
        Returns:
            DatasetDict: featurized dataset
        """
        train_text, train_labels = list(
            zip(
                *map(
                    lambda d: (d[self.text_col], d[self.label_col]), self.dataset["train"]
                )
            )
        )
        test_text, test_labels = list(
            zip(
                *map(
                    lambda d: (d[self.text_col], d[self.label_col]), self.dataset["test"]
                )
            )
        )
        dev_text, dev_labels = list(
            zip(
                *map(
                    lambda d: (d[self.text_col], d[self.label_col]), self.dataset["test"]
                )
            )
        )

        train_features = self.cv.fit_transform(train_text)
        test_features = self.cv.transform(test_text)
        dev_features = self.cv.transform(dev_text)

        # '.toarray' is needed because pyarrow doesn't support sparse matrices
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"features": train_features.toarray(), "labels": train_labels}
                ),
                "test": Dataset.from_dict(
                    {"features": test_features.toarray(), "labels": test_labels}
                ),
                "validation": Dataset.from_dict(
                    {"features": dev_features.toarray(), "labels": dev_labels}
                ),
            }
        )
        dataset.set_format(type='torch', columns=["features", "labels"])
        logging.info("LrFeaturizer: data successfully featurized and Dataset created.")
        return dataset

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
            DataLoader: training dataloader
        """
        return DataLoader(
            self.train, batch_size=self.train_batch_size, drop_last=True, num_workers=0
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: test dataloader
        """
        return DataLoader(
            self.test, batch_size=self.eval_batch_size, drop_last=True, num_workers=0
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val, batch_size=self.eval_batch_size, drop_last=True, num_workers=0
        )
