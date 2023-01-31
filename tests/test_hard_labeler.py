import omegaconf
import pytest
import torch
from hydra.utils import instantiate
from pkg_resources import resource_filename
from transformers import BertForSequenceClassification


@pytest.fixture
def conf():
    path = resource_filename("bert_squeeze", "../tests/fixtures/resources/dummy_hard_labeler_config.yaml")
    return omegaconf.OmegaConf.load(path)


class TestHardLabeler:
    def test_instantiation(self, conf):
        """"""
        try:
            instantiate(conf, _recursive_=True)
        except Exception as e:
            assert False, f"Instantiation of 'HardLabeler' raised:\n{e}"

    def test_model_tokenizer(self, conf):
        """"""
        hard_labeler = instantiate(conf, _recursive_=True).hard_labeler
        assert isinstance(hard_labeler.model, BertForSequenceClassification)

        assert hard_labeler.tokenizer.name_or_path == "bert-base-uncased"

    def test_data(self, conf):
        """"""
        hard_labeler = instantiate(conf, _recursive_=True).hard_labeler
        dataloader = hard_labeler.get_dataloader()
        batch = next(iter(dataloader))
        assert all([isinstance(v, torch.Tensor) for v in batch.values()])
        assert all([v.shape[0] == 32 for v in batch.values()])
