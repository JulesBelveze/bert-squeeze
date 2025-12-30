from typing import TypeVar

from .lm_scorer import LMScorer, SummarizationScorer
from .sequence_classification_scorer import (
    BaseSequenceClassificationScorer,
    FastBertSequenceClassificationScorer,
    LooseSequenceClassificationScorer,
)
from .seq2seq_scorer import LightningSeq2SeqScorer, Seq2SeqScorer

Scorer = TypeVar(
    "Scorer",
    LMScorer,
    SummarizationScorer,
    BaseSequenceClassificationScorer,
    FastBertSequenceClassificationScorer,
    LooseSequenceClassificationScorer,
    Seq2SeqScorer,
    LightningSeq2SeqScorer,
)
