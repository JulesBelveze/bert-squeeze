import pytest
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from bert_squeeze.assistants.train_assistant import TrainAssistant
from bert_squeeze.data.modules import LrDataModule, LSTMDataModule, TransformerDataModule
from bert_squeeze.models import (
    BowLogisticRegression,
    LtCustomBert,
    LtDeeBert,
    LtFastBert,
    LtLSTM,
    LtTheseusBert,
)


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
        assert isinstance(bert_assistant.model, LtCustomBert)
        assert bert_assistant.model.encoder.config._name_or_path == "bert-base-uncased"
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
