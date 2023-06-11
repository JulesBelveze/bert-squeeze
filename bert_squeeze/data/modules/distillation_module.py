from typing import Optional, Union

import datasets
import lightning.pytorch as pl
from datasets import Features
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ...distillation.utils.labeler import HardLabeler
from .lr_module import LrDataModule
from .lstm_module import LSTMDataModule
from .transformer_module import TransformerDataModule

TeacherDataModule = Union[LrDataModule, TransformerDataModule, LSTMDataModule]
StudentDataModule = Union[LrDataModule, TransformerDataModule, LSTMDataModule]


class DistillationDataModule(pl.LightningDataModule):
    """
    LightningDataModule for Distillation procedures.

    The module stitches together the teacher and student DataModules as they
    might require different processing steps depending on the models used.

    The module features two extra options:
    - It can generate extra labeled samples by using an `HardLabeler`.
    - It accepts an additional dataset that will be softly labeled by the
      teacher and used to further distil its knowledge.

    Args:
        teacher_module (TeacherDataModule):
            LightningDataModule to use for the teacher model.
        student_module (StudentDataModule):
            LightningDataModule to use for the student model.
        soft_data_config (DictConfig):
            Configuration to use for the "soft" dataset.
        hard_labeler (HardLabeler):
            Instance of HardLabeler to use to generate hardly labeled samples.
    """

    def __init__(
        self,
        teacher_module: TeacherDataModule,
        student_module: StudentDataModule,
        soft_data_config: DictConfig = None,
        hard_labeler: HardLabeler = None,
        **kwargs,
    ):
        super().__init__()
        assert student_module.dataset_config.path == teacher_module.dataset_config.path
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

    @staticmethod
    def _concat_dataset(
        a: Union[datasets.Dataset, datasets.DatasetDict],
        b: Union[datasets.Dataset, datasets.DatasetDict],
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """"""
        assert type(a) == type(b) and a.keys() == b.keys()

        if isinstance(a, datasets.DatasetDict):
            concat_dataset = datasets.DatasetDict(
                {
                    key: datasets.concatenate_datasets([a[key], b[key]], axis=1)
                    for key in a.keys()
                }
            )
        else:
            concat_dataset = datasets.concatenate_datasets([a, b], axis=1)
        return concat_dataset

    def create_hard_dataset(self) -> datasets.Dataset:
        """"""
        hard_dataset = self.labeler.label_dataset()
        return datasets.Dataset.from_dict(hard_dataset)

    def get_soft_dataset(self) -> datasets.Dataset:
        """
        Loads and adds a "fake label" to the dataset that will later be used
        for soft distillation.

        Returns:
            datasets.DatasetDict: dataset used for soft distillation
        """

        def _create_fake_label(example):
            example["label"] = -100
            return example

        soft_dataset = datasets.load_dataset(
            self.soft_dataset_config.path, self.soft_dataset_config.split
        )

        if self.soft_dataset_config.text_col != "text":
            soft_dataset = soft_dataset.rename_column(
                self.soft_dataset_config.text_col, "text"
            )
            self.soft_dataset_config.text_col = "text"

        # adding a "fake label" to the soft dataset for consistency with the labeled one
        max_samples = self.soft_dataset_config.get(
            "max_samples", len(soft_dataset["train"])
        )
        soft_dataset = soft_dataset["train"].select(range(max_samples))
        soft_dataset = soft_dataset.map(
            lambda example: _create_fake_label(example), batched=False
        )

        columns_to_remove = list(set(soft_dataset.column_names) - {"text", "label"})

        soft_dataset = soft_dataset.remove_columns(columns_to_remove)
        soft_dataset.features["label"] = datasets.Value(
            "null"
        )  # hack for concatenation purposes
        return soft_dataset

    def featurize(self) -> datasets.DatasetDict:
        """
        Method to featurize both teacher and student data and to concatenate them
        into one dataset.

        Returns:
            datasets.DatasetDict: concatenated dataset ready to be used for distillation
        """
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
            columns_to_keep = {"labels", "input_ids", "token_type_ids", "attention_mask"}

            features_teacher = teacher_data["train"].features.copy()
            features_teacher = Features(
                {
                    key: value
                    for key, value in features_teacher.items()
                    if key in columns_to_keep
                }
            )
            features_student = student_data["train"].features.copy()
            features_student = Features(
                {
                    key: value
                    for key, value in features_student.items()
                    if key in columns_to_keep
                }
            )

            hardly_labeled_dataset = self.create_hard_dataset()

            hardly_labeled_dataset = hardly_labeled_dataset.cast(features_teacher)
            teacher_data["train"] = datasets.concatenate_datasets(
                [teacher_data["train"], hardly_labeled_dataset]
            )

            hardly_labeled_dataset = hardly_labeled_dataset.cast(features_student)
            student_data["train"] = datasets.concatenate_datasets(
                [student_data["train"], hardly_labeled_dataset]
            )

        for dataset_split, columns in teacher_data.column_names.items():
            for col in columns:
                teacher_data[dataset_split] = teacher_data[dataset_split].rename_column(
                    col, "t_" + col
                )

        for dataset_split, columns in student_data.column_names.items():
            for col in columns:
                student_data[dataset_split] = student_data[dataset_split].rename_column(
                    col, "s_" + col
                )

        # Merging the student and teacher datasets into a single one
        concat_dataset = self._concat_dataset(teacher_data, student_data)
        concat_dataset = concat_dataset.shuffle()
        concat_dataset.set_format(type="torch")
        return concat_dataset

    def prepare_data(self) -> None:
        """
        Load teacher and student datasets
        """
        self.teacher_module.prepare_data()
        self.student_module.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """"""
        featurized_dataset = self.featurize()

        self.train = featurized_dataset["train"]
        self.val = (
            featurized_dataset["validation"]
            if "validation" in featurized_dataset.keys()
            else featurized_dataset["test"]
        )
        self.test = featurized_dataset["test"]

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: Train dataloader
        """
        return DataLoader(self.train, batch_size=self.train_batch_size, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(self.test, batch_size=self.eval_batch_size, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(self.val, batch_size=self.eval_batch_size, drop_last=True)
