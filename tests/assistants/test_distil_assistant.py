import torch.nn
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from bert_squeeze.assistants.distil_assistant import DistilAssistant


class TestDistilAssistant:
    """"""

    def test_two_hf_models(self):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            dataset_path=resource_filename("bert_squeeze", "data/local_datasets/parallel_dataset.py")
        )
        assert distil_assistant.teacher is None
        assert distil_assistant.student is None

        _ = distil_assistant.model

        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_student_torch_model(self, lr_model):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            dataset_path=resource_filename("bert_squeeze", "data/local_datasets/parallel_dataset.py"),
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            }
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_teacher_torch_model(self, lr_model):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            dataset_path=resource_filename("bert_squeeze", "data/local_datasets/parallel_dataset.py"),
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            }
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_torch_models(self, lr_model):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            dataset_path=resource_filename("bert_squeeze", "data/local_datasets/parallel_dataset.py"),
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin"
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            }
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_data(self):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            dataset_path=resource_filename("bert_squeeze", "data/local_datasets/parallel_dataset.py"),
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin"
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
            data_kwargs={
                "train_batch_size": 16,
                "eval_batch_size": 4
            }
        )
        assert isinstance(distil_assistant.data.train_dataloader(), DataLoader)
        assert len(distil_assistant.data.train_dataloader()) == 6
        assert len(distil_assistant.data.val_dataloader()) == 25
