from typing import Optional

import datasets
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from pkg_resources import resource_filename
from torch.utils.data import DataLoader

from ...distillation.utils.labeler import HardLabeler


class DistillationDataModule(pl.LightningDataModule):
    """DataModule for Distillation procedures."""

    def __init__(self, teacher_module: pl.LightningDataModule, student_module: pl.LightningDataModule,
                 soft_data_config: HydraConfig = None, hard_labeler: HardLabeler = None, **kwargs):
        super().__init__()
        assert student_module.dataset_config.name == teacher_module.dataset_config.name
        self.teacher_module = teacher_module
        self.student_module = student_module

        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)

        self.soft_dataset_config = soft_data_config
        self.labeler = hard_labeler

        self.dataset = None
        self.train = None
        self.test = None
        self.val = None

    def create_hard_dataset(self):
        """"""
        hard_dataset = self.labeler.label_dataset()
        return datasets.Dataset.from_dict(hard_dataset)

    def get_soft_dataset(self) -> datasets.Dataset:
        """"""

        def _create_fake_label(example):
            example["label"] = -100
            return example

        if self.soft_dataset_config.is_local:
            soft_dataset = datasets.load_dataset(
                resource_filename("bert-squeeze", f"data/datasets/{self.soft_dataset_config.name}_dataset.py"),
                self.soft_dataset_config.split
            )
        else:
            soft_dataset = datasets.load_dataset(self.soft_dataset_config.name, self.soft_dataset_config.split)

        if self.soft_dataset_config.text_col != "text":
            soft_dataset.rename_column_(self.soft_dataset_config.text_col, "text")
            self.soft_dataset_config.text_col = "text"

        # adding a "fake label" to the soft dataset for consistency with the labeled one
        max_samples = self.soft_dataset_config.get("max_samples", len(soft_dataset["train"]))
        soft_dataset = soft_dataset["train"].select(range(max_samples))
        soft_dataset = soft_dataset.map(lambda example: _create_fake_label(example), batched=False)

        columns_to_remove = list(set(soft_dataset.column_names) - {"text", "label"})
        return soft_dataset.remove_columns(columns_to_remove)

    def featurize(self) -> datasets.DatasetDict:
        """Featurize dataset"""

        # In case of a soft distillation
        if self.soft_dataset_config is not None:
            soft_dataset = self.get_soft_dataset()
            # Overriding the teacher & students datasets to integrate the soft dataset
            # Note: this is only done for the train set.
            self.teacher_module.dataset["train"] = datasets.concatenate_datasets(
                [self.teacher_module.dataset["train"], soft_dataset]
            )
            self.student_module.dataset["train"] = datasets.concatenate_datasets(
                [self.student_module.dataset["train"], soft_dataset]
            )

        teacher_data = self.teacher_module.featurize()
        student_data = self.student_module.featurize()

        # In case of a hard distillation
        if self.labeler is not None:
            hardly_labeled_dataset = self.create_hard_dataset()
            teacher_data["train"] = datasets.concatenate_datasets([teacher_data["train"], hardly_labeled_dataset])
            student_data["train"] = datasets.concatenate_datasets([student_data["train"], hardly_labeled_dataset])

        for dataset_split, columns in teacher_data.column_names.items():
            for col in columns:
                teacher_data[dataset_split] = teacher_data[dataset_split].rename_column(col, "t_" + col)

        for dataset_split, columns in student_data.column_names.items():
            for col in columns:
                student_data[dataset_split] = student_data[dataset_split].rename_column(col, "s_" + col)

        # Merging the student and teacher datasets into a single one
        concat_dataset = datasets.DatasetDict({
            key: datasets.concatenate_datasets(
                [teacher_data[key], student_data[key]],
                axis=1
            )
            for key in ["train", "test", "validation"]
        })
        concat_dataset = concat_dataset.shuffle()
        concat_dataset.set_format(type="torch")
        return concat_dataset

    def prepare_data(self) -> None:
        """Load and featurize dataset"""
        self.teacher_module.prepare_data()
        self.student_module.prepare_data()

    def setup(self, stage: Optional[str] = None):
        featurized_dataset = self.featurize()

        self.train = featurized_dataset["train"]
        self.val = featurized_dataset["validation"]
        self.test = featurized_dataset["test"]

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader"""
        return DataLoader(self.train, batch_size=self.train_batch_size, drop_last=True, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(self.test, batch_size=self.eval_batch_size, drop_last=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        return DataLoader(self.val, batch_size=self.eval_batch_size, drop_last=True, num_workers=0)
