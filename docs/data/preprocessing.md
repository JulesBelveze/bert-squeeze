# Seq2Seq Preprocessing Utilities

`Seq2SeqPreprocessor` offers light-touch cleaning and filtering for HuggingFace datasets
used across summarization, translation, and other generation tasks. It intentionally
focuses on common needs so you can compose the class with additional project-specific
logic when required.

## Features

- HTML/URL stripping, whitespace normalization, optional special-token removal.
- Custom cleaner hooks that run after the built-in steps.
- Word/character length filtering with independent bounds for source and target fields.
- Dataset-aware `create_splits` helper that works with `datasets.Dataset`.

## Usage

```python
from datasets import load_dataset
from bert_squeeze.data.preprocessing import Seq2SeqPreprocessor

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
preprocessor = Seq2SeqPreprocessor(
    source_field="article",
    target_field="highlights",
    filter_by_length=True,
    min_source_length=50,
    min_target_length=5,
)

clean_dataset = preprocessor.process(dataset, num_proc=4)
splits = Seq2SeqPreprocessor.create_splits(clean_dataset, train_size=0.9, val_size=0.05, test_size=0.05)
```

### Exposed Helpers

You can import the cleaning utilities directly for ad-hoc data munging:

```python
from bert_squeeze.data.preprocessing import remove_html_tags, remove_urls, normalize_whitespace
```

### Tips

- Leave `filter_by_length=False` (default) if you only need cleaning.
- All cleaning steps are optional. Pass `clean_text=False` to skip everything except
  custom cleaners.
- `create_splits` gracefully handles zero-sized validation or test ratios when you only
  need two-way splits.
