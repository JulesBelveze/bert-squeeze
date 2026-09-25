import datasets
import pytest
from omegaconf import OmegaConf

from bert_squeeze.data.modules.transformer_module import TransformerDataModule


class _Tokenizer:
    def __init__(self, name_or_path: str, emits_token_type_ids: bool):
        self.name_or_path = name_or_path
        self._emits_token_type_ids = emits_token_type_ids

    def __call__(self, text, **kwargs):
        encoding = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        if self._emits_token_type_ids:
            encoding["token_type_ids"] = [0, 0, 0]
        return encoding


@pytest.fixture
def make_module(monkeypatch):
    def _make(name_or_path: str = "bert-base-cased", emits_token_type_ids: bool = True):
        monkeypatch.setattr(
            "bert_squeeze.data.modules.transformer_module.AutoTokenizer",
            type(
                "MockTokenizer",
                (),
                {"from_pretrained": staticmethod(lambda *_, **__: object())},
            ),
        )
        config = OmegaConf.create({"text_col": "text", "label_col": "label"})
        module = TransformerDataModule(config, tokenizer_name="mock", max_length=8)
        module.tokenizer = _Tokenizer(name_or_path, emits_token_type_ids)
        return module

    return _make


def test_encode_labels_converts_string_column(make_module):
    module = make_module()
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"text": ["a", "b", "c"], "label": ["news", "research", "news"]}
            )
        }
    )

    encoded = module._encode_labels(dataset)

    assert isinstance(encoded["train"].features["label"], datasets.ClassLabel)
    # class_encode_column sorts the label names alphabetically: news=0, research=1
    assert encoded["train"]["label"] == [0, 1, 0]


def test_encode_labels_applies_label_map_and_drops_unmapped(make_module):
    module = make_module()
    module.dataset_config.label_map = {"a": "X", "b": "Y"}  # "c" is dropped
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"text": ["1", "2", "3"], "label": ["a", "b", "c"]}
            )
        }
    )

    encoded = module._encode_labels(dataset)

    assert len(encoded["train"]) == 2  # unmapped "c" dropped
    feature = encoded["train"].features["label"]
    assert isinstance(feature, datasets.ClassLabel)
    assert set(feature.names) == {"X", "Y"}


def test_encode_labels_noop_for_int_column(make_module):
    module = make_module()
    dataset = datasets.DatasetDict(
        {"train": datasets.Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})}
    )

    encoded = module._encode_labels(dataset)

    assert encoded["train"]["label"] == [0, 1]
    assert not isinstance(encoded["train"].features["label"], datasets.ClassLabel)


def test_featurize_omits_token_type_ids_when_absent(make_module):
    module = make_module(
        name_or_path="answerdotai/ModernBERT-base", emits_token_type_ids=False
    )
    module.dataset = datasets.DatasetDict(
        {
            split: datasets.Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
            for split in ("train", "validation", "test")
        }
    )

    featurized = module.featurize()

    columns = featurized["train"].format["columns"]
    assert "token_type_ids" not in columns
    assert {"input_ids", "attention_mask", "labels"}.issubset(columns)


def test_featurize_keeps_token_type_ids_when_present(make_module):
    module = make_module(name_or_path="bert-base-cased", emits_token_type_ids=True)
    module.dataset = datasets.DatasetDict(
        {
            split: datasets.Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
            for split in ("train", "validation", "test")
        }
    )

    featurized = module.featurize()

    assert "token_type_ids" in featurized["train"].format["columns"]
