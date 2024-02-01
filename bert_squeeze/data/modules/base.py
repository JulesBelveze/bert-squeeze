import logging

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
        self.dataset = datasets.load_dataset(self.dataset_config.path)

        if "percent" not in self.dataset_config:
            return

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].select(
                range(int(len(self.dataset[split]) * self.dataset_config.percent / 100))
            )

        logging.info(f"DatasetDict '{self.dataset_config.path}' successfully loaded.")
