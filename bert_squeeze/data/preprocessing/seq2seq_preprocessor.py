from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from datasets import Dataset, DatasetDict

Cleaner = Callable[[str], str]

_URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
_HTML_REGEX = re.compile(r"<[^>]+>")


def remove_html_tags(text: str) -> str:
    return _HTML_REGEX.sub(" ", text)


def remove_urls(text: str) -> str:
    return _URL_REGEX.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_special_tokens(tokens: str, specials: Sequence[str]) -> str:
    cleaned = tokens
    for token in specials:
        cleaned = cleaned.replace(token, " ")
    return normalize_whitespace(cleaned)


class Seq2SeqPreprocessor:
    """Lightweight text cleaning and filtering pipeline for seq2seq datasets."""

    def __init__(
        self,
        source_field: str,
        target_field: str,
        *,
        clean_text: bool = True,
        remove_html: bool = True,
        remove_urls_flag: bool = True,
        normalize_spaces: bool = True,
        special_tokens: Optional[Sequence[str]] = None,
        filter_by_length: bool = False,
        min_source_length: Optional[int] = None,
        max_source_length: Optional[int] = None,
        min_target_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        length_unit: str = "words",
        custom_cleaners: Optional[Iterable[Cleaner]] = None,
    ) -> None:
        self.source_field = source_field
        self.target_field = target_field
        self.clean_text = clean_text
        self.filter_by_length = filter_by_length
        self.length_unit = length_unit
        self.min_source_length = min_source_length
        self.max_source_length = max_source_length
        self.min_target_length = min_target_length
        self.max_target_length = max_target_length

        self.cleaners: List[Cleaner] = []
        if clean_text:
            if remove_html:
                self.cleaners.append(remove_html_tags)
            if remove_urls_flag:
                self.cleaners.append(remove_urls)
            if normalize_spaces:
                self.cleaners.append(normalize_whitespace)
            if special_tokens:
                self.cleaners.append(
                    lambda text: remove_special_tokens(text, special_tokens)
                )
            if custom_cleaners:
                self.cleaners.extend(custom_cleaners)

    def process(
        self,
        dataset: Union[Dataset, DatasetDict],
        *,
        num_proc: Optional[int] = None,
    ) -> Union[Dataset, DatasetDict]:
        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    split: self._process_dataset(split_ds, num_proc=num_proc)
                    for split, split_ds in dataset.items()
                }
            )
        return self._process_dataset(dataset, num_proc=num_proc)

    def _process_dataset(
        self,
        dataset: Dataset,
        *,
        num_proc: Optional[int],
    ) -> Dataset:
        processed = dataset
        if self.cleaners:
            processed = processed.map(
                self._clean_example,
                num_proc=num_proc,
            )
        if self.filter_by_length:
            processed = processed.filter(self._length_filter, num_proc=num_proc)
        return processed

    def _clean_example(self, example: Dict[str, str]) -> Dict[str, str]:
        example = example.copy()
        example[self.source_field] = self._run_cleaners(example[self.source_field])
        example[self.target_field] = self._run_cleaners(example[self.target_field])
        return example

    def _run_cleaners(self, text: str) -> str:
        cleaned = text
        for cleaner in self.cleaners:
            cleaned = cleaner(cleaned)
        return cleaned

    def _length_filter(self, example: Dict[str, str]) -> bool:
        src_len = self._length_of(example[self.source_field])
        tgt_len = self._length_of(example[self.target_field])
        return self._length_in_range(
            src_len, self.min_source_length, self.max_source_length
        ) and self._length_in_range(
            tgt_len, self.min_target_length, self.max_target_length
        )

    def _length_of(self, text: str) -> int:
        if self.length_unit == "words":
            return len(text.split())
        if self.length_unit == "chars":
            return len(text)
        raise ValueError(
            f"Unsupported length unit '{self.length_unit}'. Use 'words' or 'chars'."
        )

    @staticmethod
    def _length_in_range(
        value: int,
        min_value: Optional[int],
        max_value: Optional[int],
    ) -> bool:
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    @staticmethod
    def create_splits(
        dataset: Dataset,
        *,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
        stratify_by_column: Optional[str] = None,
    ) -> DatasetDict:
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.")
        if not isinstance(dataset, Dataset):
            raise TypeError("create_splits expects a datasets.Dataset input.")

        first_split = dataset.train_test_split(
            train_size=train_size,
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
        train_dataset = first_split["train"]
        remainder = first_split["test"]

        if val_size == 0 and test_size == 0:
            return DatasetDict({"train": train_dataset})
        if val_size == 0:
            return DatasetDict({"train": train_dataset, "test": remainder})
        if test_size == 0:
            return DatasetDict({"train": train_dataset, "validation": remainder})

        val_ratio = val_size / (val_size + test_size)
        val_test_split = remainder.train_test_split(
            train_size=val_ratio,
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
        return DatasetDict(
            {
                "train": train_dataset,
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            }
        )
