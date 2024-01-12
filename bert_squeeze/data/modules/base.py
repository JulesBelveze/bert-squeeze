import logging

import datasets
import lightning.pytorch as pl


class BaseDataModule(pl.LightningDataModule):
    def prepare_data(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_config.path)

        if "percent" not in self.dataset_config:
            return

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].select(
                range(int(len(self.dataset[split]) * self.dataset_config.percent / 100))
            )

        logging.info(f"DatasetDict '{self.dataset_config.path}' successfully loaded.")
