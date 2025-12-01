import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from bert_squeeze.assistants.train_assistant import TrainAssistant
from bert_squeeze.data.modules import LrDataModule, LSTMDataModule, TransformerDataModule
from bert_squeeze.models import (
    BowLogisticRegression,
    LtAdapter,
    LtBerxit,
    LtDeeBert,
    LtFastBert,
    LtLSTM,
    LtSequenceClassificationCustomBert,
    LtTheseusBert,
)
from bert_squeeze.models.custom_transformers import BaseEncoderDecoderModel


@pytest.fixture
def lr_assistant():
    return TrainAssistant(
        "lr",
        data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
        general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
    )


class TestTrainAssistant:
    def test_sanity_assistant(self, lr_assistant):
        """"""
        assert lr_assistant.general.num_labels == 6
        assert isinstance(lr_assistant.model, BowLogisticRegression)
        assert isinstance(lr_assistant.data, LrDataModule)
        assert isinstance(lr_assistant.logger, TensorBoardLogger)

    def test_data(self, lr_assistant):
        """"""
        assert isinstance(lr_assistant.data.train_dataloader(), DataLoader)
        assert len(lr_assistant.data.train_dataloader()) == 50

    def test_bert_assistant(self):
        """"""
        bert_assistant = TrainAssistant(
            "bert",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={"pretrained_model": "bert-base-uncased"},
        )
        assert bert_assistant.general.num_labels == 6
        assert isinstance(bert_assistant.model, LtSequenceClassificationCustomBert)
        assert bert_assistant.model.model.config._name_or_path == "bert-base-uncased"
        assert isinstance(bert_assistant.data, TransformerDataModule)

    def test_lstm_assistant(self):
        """"""
        lstm_assistant = TrainAssistant(
            "lstm",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
        )
        assert lstm_assistant.general.num_labels == 6
        assert isinstance(lstm_assistant.model, LtLSTM)
        assert isinstance(lstm_assistant.data, LSTMDataModule)

    def test_deebert_assistant(self):
        """"""
        deebert_assistant = TrainAssistant(
            "deebert",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={"pretrained_model": "bert-base-uncased"},
        )
        assert deebert_assistant.general.num_labels == 6
        assert isinstance(deebert_assistant.model, LtDeeBert)
        assert deebert_assistant.model.bert.config._name_or_path == "bert-base-uncased"
        assert isinstance(deebert_assistant.data, TransformerDataModule)

    def test_berxit_assistant(self):
        """"""
        berxit_assistant = TrainAssistant(
            "berxit",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={"pretrained_model": "bert-base-uncased"},
        )
        assert berxit_assistant.general.num_labels == 6
        assert isinstance(berxit_assistant.model, LtBerxit)
        assert berxit_assistant.model.bert.config._name_or_path == "bert-base-uncased"
        assert isinstance(berxit_assistant.data, TransformerDataModule)

    def test_fastbert_assistant(self):
        """"""
        fastbert_assistant = TrainAssistant(
            "fastbert",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={"pretrained_model": "bert-base-uncased"},
        )
        assert fastbert_assistant.general.num_labels == 6
        assert isinstance(fastbert_assistant.model, LtFastBert)
        assert (
            fastbert_assistant.model.encoder.config._name_or_path == "bert-base-uncased"
        )
        assert isinstance(fastbert_assistant.data, TransformerDataModule)
        assert len(fastbert_assistant.callbacks) > 0

    def test_theseusbert_assistant(self):
        """"""
        fastbert_assistant = TrainAssistant(
            "theseusbert",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={"pretrained_model": "bert-base-uncased"},
        )
        assert fastbert_assistant.general.num_labels == 6
        assert isinstance(fastbert_assistant.model, LtTheseusBert)
        assert (
            fastbert_assistant.model.encoder.config._name_or_path == "bert-base-uncased"
        )
        assert isinstance(fastbert_assistant.data, TransformerDataModule)

    def test_adapter_assistant(self):
        """"""
        adapter_assistant = TrainAssistant(
            "adapter",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={
                "pretrained_model": "bert-base-uncased",
                "task_name": "setfit",
                "labels": [0, 1, 2, 3, 4, 5],
            },
        )
        assert adapter_assistant.general.num_labels == 6
        assert isinstance(adapter_assistant.model, LtAdapter)
        assert isinstance(adapter_assistant.data, TransformerDataModule)

    def test_adapter_training(self):
        """"""
        adapter_assistant = TrainAssistant(
            "adapter",
            data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 5}},
            general_kwargs={"labels": [0, 1, 2, 3, 4, 5], "num_labels": 6},
            model_kwargs={
                "pretrained_model": "bert-base-uncased",
                "task_name": "setfit",
                "labels": [0, 1, 2, 3, 4, 5],
            },
        )
        model = adapter_assistant.model

        train_dataloader = adapter_assistant.data.train_dataloader()
        test_dataloader = adapter_assistant.data.test_dataloader()

        basic_trainer = Trainer(max_steps=4)
        basic_trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
        )


class TestSeq2SeqTraining:
    def test_t5_summarization(self):
        """"""
        adapter_assistant = TrainAssistant(
            "t5",
            data_kwargs={
                "dataset_config": {
                    "path": "kmfoda/booksum",
                    "percent": 5,
                    "target_col": "summary",
                    "source_col": "chapter",
                }
            },
            model_kwargs={
                "pretrained_model": "t5-small",
                "task": "summarization",
                "scorer": {
                    "_target_": "bert_squeeze.utils.scorers.lm_scorer.SummarizationScorer",
                    "tokenizer_name": "t5-small",
                },
            },
        )
        model = adapter_assistant.model

        train_dataloader = adapter_assistant.data.train_dataloader()
        test_dataloader = adapter_assistant.data.test_dataloader()
        basic_trainer = Trainer(max_steps=4)
        basic_trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
        )

    def test_t5_summarization_2(self):
        """"""
        adapter_assistant = TrainAssistant(
            "t5",
            data_kwargs={
                "dataset_config": {
                    "path": "kmfoda/booksum",
                    "percent": 5,
                    "target_col": "summary",
                    "source_col": "chapter",
                }
            },
            model_kwargs={
                "model": {
                    "_target_": "bert_squeeze.models.custom_transformers.BaseEncoderDecoderModel",
                    "model": {
                        "_target_": "transformers.models.t5.T5ForConditionalGeneration.from_pretrained",
                        "pretrained_model_name_or_path": "t5-small",
                    },
                },
                "pretrained_model": "t5-small",
                "task": "summarization",
                "scorer": {
                    "_target_": "bert_squeeze.utils.scorers.lm_scorer.SummarizationScorer",
                    "tokenizer_name": "t5-small",
                },
            },
        )

        train_dataloader = adapter_assistant.data.train_dataloader()
        test_dataloader = adapter_assistant.data.test_dataloader()

        basic_trainer = Trainer(max_steps=4)
        basic_trainer.fit(
            model=adapter_assistant.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
        )
