# Seq2Seq Evaluation Metrics

`Seq2SeqScorer` extends the existing scorer toolbox with metrics that are typically
required for summarization, translation, paraphrasing, and other generation tasks.
Under the hood it leverages the [`evaluate`](https://github.com/huggingface/evaluate)
package so the scorer is drop-in for both research notebooks and Lightning modules.

## Requirements

The core dependencies already include the required metric packages. If you rely on
extras-based installs, make sure to pull the `seq2seq` extra:

```bash
uv pip install "bert-squeeze[seq2seq]"
# or, when developing inside the repo:
uv sync
```

## Basic Usage

```python
from bert_squeeze.utils.scorers import Seq2SeqScorer

predictions = ["tiny summary", "concise second summary"]
references = ["tiny summary", "concise reference"]

scorer = Seq2SeqScorer(metrics=["rouge", "bleu", "meteor"])
scores = scorer.compute(predictions, references)

print(scores["rouge1"])  # => 0.67
print(scores["bleu"])    # => 58.3
```

### Group-Based Evaluation

Pass `group_labels` to `compute` to slice scores by language, domain, dataset shard, or
any other label. Each group receives a `{group}_{metric}` prefix.

```python
scores = scorer.compute(
    predictions=preds,
    references=refs,
    group_labels=batch["language"],
)
print(scores["en_rougeL"])  # => English-only ROUGE-L
```

### Length Statistics

Length comparison metrics are available out of the box—average lengths, ratios, and mean
absolute error—computed in words by default. Toggle them with
`compute_length_stats=False` or switch to character-based stats via `length_unit="chars"`.

### Lightning Integration

`LightningSeq2SeqScorer` wraps `Seq2SeqScorer` and returns ready-to-log dictionaries:

```python
from bert_squeeze.utils.scorers import LightningSeq2SeqScorer

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.val_scorer = LightningSeq2SeqScorer(
            prefix="val",
            metrics=["rouge", "bertscore"],
            bertscore_kwargs={"lang": "en", "model_type": "microsoft/deberta-base-mnli"},
        )

    def validation_step(self, batch, _):
        predictions = self.generate_text(batch["input_ids"])
        references = batch["labels_text"]
        scores = self.val_scorer.get_log_dict(
            predictions,
            references,
            group_labels=batch.get("language"),
        )
        self.log_dict(scores, prog_bar=True, sync_dist=True)
```

### Tips

- Metric modules are lazily loaded and cached, so you pay initialization cost only when a
  metric is first used.
- BERTScore accepts a custom `bertscore_kwargs` dict for multilingual models (e.g.,
  `{"lang": "fr", "model_type": "xlm-roberta-large"}`).
- Group labels are sanitized to lowercase snake case so they remain logging-friendly.
