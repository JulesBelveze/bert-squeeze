from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from bert_squeeze.distillation.seq2seq_distiller import Seq2SeqDistiller
from bert_squeeze.distillation.sequence_classification_distiller import (
    SequenceClassificationDistiller,
    SequenceClassificationParallelDistiller,
)
from bert_squeeze.utils.losses import KLDivLoss


def _loss_context() -> SimpleNamespace:
    return SimpleNamespace(
        loss_ce=torch.nn.CrossEntropyLoss(),
        loss_distill=KLDivLoss(T=2.0),
        params=SimpleNamespace(alpha=0.0),
    )


def _seq2seq_logits() -> tuple[torch.Tensor, torch.Tensor]:
    teacher_logits = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, 1.5, -0.5]], [[-1.0, 0.0, 1.0], [1.0, -0.5, 0.0]]]
    )
    student_logits = torch.tensor(
        [[[0.0, 1.0, -1.0], [1.0, 0.5, -0.5]], [[0.5, -1.0, 1.0], [0.0, 1.0, -0.5]]],
        requires_grad=True,
    )
    return teacher_logits, student_logits


def test_kl_div_loss_matches_pytorch_batch_mean_over_vocabulary():
    teacher_logits, student_logits = _seq2seq_logits()
    temperature = 2.0

    loss = KLDivLoss(T=temperature)(student_logits, teacher_logits)
    expected = (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )

    torch.testing.assert_close(loss, expected)


def test_sequence_classification_distiller_computes_kl_from_student_to_teacher():
    teacher_logits, student_logits = _seq2seq_logits()
    teacher_logits = teacher_logits[:, 0, :]
    student_logits = student_logits[:, 0, :]
    labels = torch.tensor([0, 1])

    loss = SequenceClassificationDistiller.loss(
        _loss_context(), teacher_logits, student_logits, labels
    )

    torch.testing.assert_close(
        loss.kd_loss, KLDivLoss(T=2.0)(student_logits, teacher_logits)
    )


def test_seq2seq_distiller_ignores_masked_tokens_for_kl_loss():
    teacher_logits, student_logits = _seq2seq_logits()
    labels = torch.tensor([[0, -100], [2, 0]])

    loss = Seq2SeqDistiller.loss(_loss_context(), teacher_logits, student_logits, labels)

    active_idx = labels != -100
    expected = KLDivLoss(T=2.0)(student_logits[active_idx], teacher_logits[active_idx])
    torch.testing.assert_close(loss.kd_loss, expected)


def test_seq2seq_distiller_all_ignored_labels_returns_a_differentiable_zero():
    teacher_logits, student_logits = _seq2seq_logits()
    labels = torch.full((2, 2), -100)

    loss = Seq2SeqDistiller.loss(_loss_context(), teacher_logits, student_logits, labels)

    assert loss.full_loss.requires_grad
    loss.full_loss.backward()
    torch.testing.assert_close(student_logits.grad, torch.zeros_like(student_logits))


def test_translation_distiller_computes_kl_from_each_student_output():
    teacher_logits, student_logits = _seq2seq_logits()
    teacher_logits = teacher_logits[:, 0, :]
    student_logits = student_logits[:, 0, :]
    student_logits_translation = student_logits.flip(dims=[-1])

    loss = SequenceClassificationParallelDistiller.loss(
        _loss_context(), teacher_logits, student_logits, student_logits_translation
    )

    expected_original = KLDivLoss(T=2.0)(student_logits, teacher_logits)
    expected_translation = KLDivLoss(T=2.0)(student_logits_translation, teacher_logits)
    torch.testing.assert_close(loss.kd_loss, expected_original)
    torch.testing.assert_close(loss.full_loss, expected_original + expected_translation)


class _Seq2SeqTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, labels):
        logits = input_ids.float().unsqueeze(-1) * self.scale
        return Seq2SeqLMOutput(logits=logits)


def test_seq2seq_teacher_logits_do_not_track_gradients():
    teacher = _Seq2SeqTeacher()
    teacher.train()
    batch = {
        "t_labels": torch.tensor([[1, 2]]),
        "t_input_ids": torch.tensor([[1, 2]]),
        "t_attention_mask": torch.tensor([[1, 1]]),
    }

    logits = Seq2SeqDistiller.get_teacher_logits(SimpleNamespace(teacher=teacher), batch)

    assert not teacher.training
    assert not logits.requires_grad
