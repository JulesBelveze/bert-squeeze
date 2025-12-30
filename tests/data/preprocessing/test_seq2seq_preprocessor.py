import pytest
from datasets import Dataset, DatasetDict

from bert_squeeze.data.preprocessing import (
    Seq2SeqPreprocessor,
    remove_html_tags,
    remove_urls,
)


def _build_dataset(inputs, targets):
    return Dataset.from_dict({"source": inputs, "target": targets})


def test_basic_cleaning_pipeline():
    dataset = _build_dataset(
        ["<p>Hello</p> visit https://example.com <pad>"],
        ["Multiple   spaces   here"],
    )
    preprocessor = Seq2SeqPreprocessor(
        source_field="source",
        target_field="target",
        special_tokens=["<pad>"],
    )

    processed = preprocessor.process(dataset)
    example = processed[0]

    assert "<p>" not in example["source"]
    assert "http" not in example["source"]
    assert "  " not in example["target"]


def test_length_filtering_words():
    dataset = _build_dataset(
        ["short text", "this sentence is long enough"],
        ["tiny", "target sentence with words"],
    )
    preprocessor = Seq2SeqPreprocessor(
        source_field="source",
        target_field="target",
        filter_by_length=True,
        min_source_length=3,
        min_target_length=3,
    )

    processed = preprocessor.process(dataset)
    assert len(processed) == 1
    assert processed[0]["source"].startswith("this sentence")


def test_custom_cleaner_applied():
    def replace_noise(text: str) -> str:
        return text.replace("NOISE", "").strip()

    dataset = _build_dataset(["with NOISE tokens"], ["clean target"])
    preprocessor = Seq2SeqPreprocessor(
        source_field="source",
        target_field="target",
        clean_text=True,
        custom_cleaners=[replace_noise],
    )

    processed = preprocessor.process(dataset)
    assert "NOISE" not in processed[0]["source"]


def test_dataset_dict_supported():
    dataset = _build_dataset(["a"], ["b"])
    dataset_dict = DatasetDict({"train": dataset, "validation": dataset})
    preprocessor = Seq2SeqPreprocessor(
        source_field="source",
        target_field="target",
    )

    processed = preprocessor.process(dataset_dict)
    assert isinstance(processed, DatasetDict)
    assert set(processed.keys()) == {"train", "validation"}


def test_create_splits_sizes():
    dataset = _build_dataset([f"text {idx}" for idx in range(100)], ["target"] * 100)
    splits = Seq2SeqPreprocessor.create_splits(
        dataset,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
        seed=0,
    )
    assert len(splits["train"]) == 70
    assert len(splits["validation"]) == 20
    assert len(splits["test"]) == 10


def test_create_splits_invalid_ratio():
    dataset = _build_dataset(["one", "two"], ["a", "b"])
    with pytest.raises(ValueError):
        Seq2SeqPreprocessor.create_splits(
            dataset,
            train_size=0.5,
            val_size=0.3,
            test_size=0.3,
        )
