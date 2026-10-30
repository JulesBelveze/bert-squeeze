from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples/selective_quantization/compare.py"
)
spec = importlib.util.spec_from_file_location("selective_example", SCRIPT)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_example_exports_and_reloads_without_hub_access(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.manual_seed(42)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=2,
        )
    )
    model.save_pretrained(checkpoint)
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "film", "a"]
        )
    }
    backend = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(checkpoint)
    data = tmp_path / "rows.jsonl"
    data.write_text('{"text":"a good film","label":1}\n{"text":"bad","label":0}\n')
    output = tmp_path / "comparison"
    command = [
        str(SCRIPT),
        str(checkpoint),
        "--data",
        str(data),
        "--output",
        str(output),
        "--exclude",
        "classifier",
    ]
    monkeypatch.setattr(sys, "argv", command)
    example.main()
    report = json.loads((output / "comparison.json").read_text())
    assert report["rows"] == 2
    full = report["variants"]["int8_all"]
    selective = report["variants"]["int8_selective"]
    assert selective["quantized_linear_modules"] == full["quantized_linear_modules"] - 1
    assert selective["reload_verified_rows_per_policy"] == 2
    assert set(selective["quality"]) == {
        "batch16_pad256",
        "batch1_pad256",
        "batch1_actual_length",
    }
    assert isinstance(
        example.load_artifact(output / "int8_selective").classifier, torch.nn.Linear
    )
    assert selective["deployment_bytes"] > 0
    original = (output / "comparison.json").read_bytes()
    with pytest.raises(SystemExit):
        example.main()
    assert (output / "comparison.json").read_bytes() == original


def test_example_rejects_bad_labels_and_module_names(tmp_path):
    data = tmp_path / "rows.jsonl"
    for payload in (
        '{"text":"good","label":true}',
        '{"text":"good","label":2}',
        '{"text":"","label":0}',
    ):
        data.write_text(payload + "\n")
        with pytest.raises(ValueError):
            example.read_rows(data, 2)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=2,
        )
    ).eval()
    with pytest.raises(ValueError, match="exclusions"):
        example.convert(model, ["classifier.weight"])
