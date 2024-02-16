from typing import TypeVar

from .lm_scorer import LMScorer, SummarizationScorer
from .sequence_classification_scorer import (
    BaseSequenceClassificationScorer,
    FastBertSequenceClassificationScorer,
    LooseSequenceClassificationScorer,
)

Scorer = TypeVar(
    "Scorer",
    LMScorer,
    SummarizationScorer,
    BaseSequenceClassificationScorer,
    FastBertSequenceClassificationScorer,
    LooseSequenceClassificationScorer,
)
