import torch.nn
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from bert_squeeze.assistants.distil_assistant import DistilAssistant
from bert_squeeze.models.lt_t5 import SimpleT5Model

class TestDistilAssistant:
    """"""

    def test_two_hf_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil",
            data_kwargs={"path": "emotion"},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
        )
        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model

        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_student_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil",
            data_kwargs={"path": "emotion"},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            student_kwargs={
                "_target_": "transformers.models.auto.AutoModelForSequenceClassification.from_pretrained",
                "pretrained_model_name_or_path": "microsoft/xtremedistil-l6-h256-uncased",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_teacher_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil",
            data_kwargs={"path": "emotion"},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_torch_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil",
            data_kwargs={"path": "emotion"},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_data(self):
        """"""
        distil_assistant = DistilAssistant(
            "distil",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
            data_kwargs={
                "teacher_module": {
                    "_target_": "bert_squeeze.data.modules.lr_module.LrDataModule",
                    "dataset_config": {
                        "is_local": False,
                        "path": "SetFit/emotion",
                        "train_batch_size": 16,
                        "eval_batch_size": 4,
                        "text_col": "text",
                        "label_col": "label",
                    },
                    "max_features": 5000,
                },
                "student_module": {
                    "_target_": "bert_squeeze.data.modules.lr_module.LrDataModule",
                    "dataset_config": {
                        "is_local": False,
                        "path": "SetFit/emotion",
                        "train_batch_size": 16,
                        "eval_batch_size": 4,
                        "text_col": "text",
                        "label_col": "label",
                    },
                    "max_features": 5000,
                },
            },
        )
        assert isinstance(distil_assistant.data.train_dataloader(), DataLoader)
        assert len(distil_assistant.data.train_dataloader()) == 500
        assert len(distil_assistant.data.val_dataloader()) == 62


class TestDistilSoftAssistant:
    """"""

    def test_two_hf_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-soft",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {
                    "is_local": False,
                    "text_col": "text",
                    "split": "train",
                    "path": "SetFit/emotion",
                },
            },
        )
        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model

        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_student_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-soft",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {
                    "is_local": False,
                    "text_col": "text",
                    "split": "train",
                    "path": "SetFit/emotion",
                },
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_teacher_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-soft",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {
                    "is_local": False,
                    "text_col": "text",
                    "split": "train",
                    "path": "SetFit/emotion",
                },
            },
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_torch_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-soft",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {
                    "is_local": False,
                    "text_col": "text",
                    "split": "train",
                    "path": "SetFit/emotion",
                },
            },
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_data(self):
        """"""
        distil_assistant = DistilAssistant(
            "distil-soft",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
            data_kwargs={
                "is_local": False,
                "path": "SetFit/emotion",
                "soft_data_config": {
                    "is_local": False,
                    "text_col": "text",
                    "split": "train[:10%]",
                    "path": "SetFit/emotion",
                },
                "train_batch_size": 16,
                "eval_batch_size": 4,
            },
        )
        assert isinstance(distil_assistant.data.train_dataloader(), DataLoader)
        assert len(distil_assistant.data.train_dataloader()) == 1625
        assert len(distil_assistant.data.val_dataloader()) == 500


class TestDistilHardAssistant:
    """"""

    def test_two_hf_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-hard",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {
                    "is_local": False,
                    "split": "split",
                    "text_col": "text",
                    "path": "SetFit/emotion",
                    "max_samples": 10,
                },
                "train_batch_size": 2,
                "eval_batch_size": 2,
            },
        )
        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model

        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_student_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-hard",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {"is_local": False, "path": "SetFit/emotion"},
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_teacher_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-hard",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {"is_local": False, "path": "SetFit/emotion"},
            },
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_torch_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-hard",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "emotion",
                "soft_data_config": {"is_local": False, "path": "SetFit/emotion"},
            },
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_data(self):
        """"""
        distil_assistant = DistilAssistant(
            "distil-hard",
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            data_kwargs={
                "path": "SetFit/emotion",
                "split": "train",
                "hard_labeler": {
                    "dataset_config": {
                        "is_local": False,
                        "split": "train[:10%]",
                        "text_col": "text",
                        "path": "SetFit/emotion",
                        "max_samples": 100,
                    }
                },
                "train_batch_size": 2,
                "eval_batch_size": 4,
            },
        )

        assert isinstance(distil_assistant.data.train_dataloader(), DataLoader)
        assert len(distil_assistant.data.train_dataloader()) == 8048
        assert len(distil_assistant.data.val_dataloader()) == 500


class TestDistilAssistantParallel:
    """"""

    def test_two_hf_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            data_kwargs={
                "path": resource_filename(
                    "bert_squeeze", "data/local_datasets/parallel_dataset.py"
                ),
                "is_local": True,
            },
        )
        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model

        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_student_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            data_kwargs={
                "path": resource_filename(
                    "bert_squeeze", "data/local_datasets/parallel_dataset.py"
                ),
                "is_local": True,
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, BertForSequenceClassification)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_teacher_torch_model(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            data_kwargs={
                "path": resource_filename(
                    "bert_squeeze", "data/local_datasets/parallel_dataset.py"
                ),
                "is_local": True,
            },
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, BertForSequenceClassification)

    def test_torch_models(self, caplog):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            data_kwargs={
                "path": resource_filename(
                    "bert_squeeze", "data/local_datasets/parallel_dataset.py"
                ),
                "is_local": True,
            },
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, torch.nn.Module)
        assert isinstance(distil_assistant.student, torch.nn.Module)

    def test_data(self):
        """"""
        distil_assistant = DistilAssistant(
            "distil-parallel",
            teacher_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
                "checkpoints": "../tests/fixtures/resources/lr_dummy.bin",
            },
            student_kwargs={
                "_target_": "tests.fixtures.dummy_models.Lr",
            },
            data_kwargs={
                "path": resource_filename(
                    "bert_squeeze", "data/local_datasets/parallel_dataset.py"
                ),
                "is_local": True,
                "train_batch_size": 16,
                "eval_batch_size": 4,
            },
        )
        assert isinstance(distil_assistant.data.train_dataloader(), DataLoader)
        assert len(distil_assistant.data.train_dataloader()) == 187
        assert len(distil_assistant.data.val_dataloader()) == 125


class TestDistilSeq2SeqAssistant:

    def test_distil_seq2seq(self, caplog):
        distil_assistant = DistilAssistant(
            "distil-seq2seq",
            data_kwargs={
                "path": "kmfoda/booksum",
                "percent": 5,
                "target_col": "summary",
                "source_col": "chapter",
            }
        )

        assert distil_assistant.teacher is None
        assert distil_assistant.student is None
        assert "The Distiller has not been instantiated" in caplog.text

        _ = distil_assistant.model
        assert isinstance(distil_assistant.teacher, SimpleT5Model)
        assert isinstance(distil_assistant.student, SimpleT5Model)
