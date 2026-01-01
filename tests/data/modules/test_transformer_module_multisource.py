from typing import List

import datasets
import pytest
from omegaconf import OmegaConf

from bert_squeeze.data.modules.transformer_module import Seq2SeqTransformerDataModule


class _DummyTokenizer:
    name_or_path = "t5-small"

    def pad(self, *_, **__):
        return {}

    def __call__(self, *_, **__):
        return {"input_ids": [], "attention_mask": []}

    def as_target_tokenizer(self):
        return self


@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "bert_squeeze.data.modules.transformer_module.AutoTokenizer",
        type(
            "MockTokenizer", (), {"from_pretrained": lambda *_, **__: _DummyTokenizer()}
        ),
    )


def _build_dataset(values: List[str]) -> datasets.DatasetDict:
    ds = datasets.Dataset.from_dict(
        {"source": values, "target": [value.upper() for value in values]}
    )
    return datasets.DatasetDict({"train": ds, "validation": ds})


def test_prepare_data_concatenate(monkeypatch):
    config = OmegaConf.create(
        {
            "source_col": "source",
            "target_col": "target",
            "data_path": ["first", "second"],
            "data_format": ["disk", "disk"],
            "combine_strategy": "concatenate",
        }
    )
    module = Seq2SeqTransformerDataModule(
        config, tokenizer_name="t5-small", max_target_length=32, max_source_length=32
    )

    ds1 = _build_dataset(["a"])
    ds2 = _build_dataset(["b", "c"])
    monkeypatch.setattr(
        Seq2SeqTransformerDataModule,
        "_load_dataset",
        lambda self, path, fmt: ds1 if path == "first" else ds2,
    )

    module.prepare_data()
    assert len(module.dataset["train"]) == 3


def test_prepare_data_interleave(monkeypatch):
    config = OmegaConf.create(
        {
            "source_col": "source",
            "target_col": "target",
            "data_path": ["p1", "p2"],
            "data_format": "disk",
            "combine_strategy": "interleave",
        }
    )
    module = Seq2SeqTransformerDataModule(
        config, tokenizer_name="t5-small", max_target_length=32, max_source_length=32
    )

    ds1 = _build_dataset(["a"])
    ds2 = _build_dataset(["b"])
    monkeypatch.setattr(
        Seq2SeqTransformerDataModule,
        "_load_dataset",
        lambda self, path, fmt: ds1 if path == "p1" else ds2,
    )

    module.prepare_data()
    assert len(module.dataset["train"]) == 2


def test_format_list_validation():
    config = OmegaConf.create(
        {
            "source_col": "source",
            "target_col": "target",
            "data_path": ["only"],
            "data_format": ["disk", "disk"],
        }
    )
    module = Seq2SeqTransformerDataModule(
        config,
        tokenizer_name="t5-small",
        max_target_length=8,
        max_source_length=8,
    )
    with pytest.raises(ValueError):
        module.prepare_data()
