import pytest
from omegaconf import OmegaConf

from bert_squeeze.data.modules.transformer_module import Seq2SeqTransformerDataModule


class _DummyTokenizer:
    name_or_path = "t5-small"
    pad_token_id = 0
    padding_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def pad(
        self,
        features,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["attention_mask"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)

        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "bert_squeeze.data.modules.transformer_module.AutoTokenizer",
        type(
            "MockTokenizer", (), {"from_pretrained": lambda *_, **__: _DummyTokenizer()}
        ),
    )


def test_collate_pads_labels_with_ignore_index():
    config = OmegaConf.create(
        {
            "source_col": "source",
            "target_col": "target",
            "path": "dummy",
        }
    )
    module = Seq2SeqTransformerDataModule(
        config, tokenizer_name="t5-small", max_target_length=8, max_source_length=8
    )
    collate = module._collate_fn()

    batch = collate(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5]},
            {"input_ids": [6], "attention_mask": [1], "labels": [7, 8, 9]},
        ]
    )

    assert batch["labels"].shape == (2, 3)
    assert batch["labels"][0, 2].item() == -100
