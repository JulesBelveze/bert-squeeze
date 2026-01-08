import math

import pytest
import torch

from bert_squeeze.utils.scorers.lm_scorer import LMScorer, SummarizationScorer
from bert_squeeze.utils.types import DistillationLoss


def test_lm_scorer_supports_loss_only():
    scorer = LMScorer(tokenizer_name=None, do_mismatch=True)
    scorer.add(loss=torch.tensor(1.0))

    assert len(scorer.losses["global"]) == 1
    assert isinstance(scorer.losses["global"][0], torch.Tensor)
    assert scorer.perplexity == pytest.approx(math.e, rel=1e-6)


def test_lm_scorer_supports_distillation_loss():
    scorer = LMScorer(tokenizer_name=None, do_mismatch=False)
    loss = DistillationLoss(
        kd_loss=torch.tensor(0.0),
        objective=torch.tensor(0.0),
        full_loss=torch.tensor(2.0),
    )
    scorer.add(loss=loss)
    assert scorer.perplexity == pytest.approx(math.exp(2.0), rel=1e-6)


def test_summarization_scorer_table_formats_scalars():
    table = SummarizationScorer.get_table({"rouge1": 12.34})
    assert "rouge1" in table
