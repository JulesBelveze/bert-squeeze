from collections import defaultdict

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers.modeling_outputs import Seq2SeqLMOutput

from bert_squeeze.distillation.seq2seq_distiller import Seq2SeqDistiller


class _TinySeq2SeqModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(1, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        logits = self.classifier(input_ids.float().unsqueeze(-1))
        return Seq2SeqLMOutput(logits=logits)


class _NoOpScorer:
    def __init__(self) -> None:
        self.losses = defaultdict(list)

    def add(self, **kwargs: object) -> None:
        return None

    def reset(self) -> None:
        self.losses.clear()


class _OfflineSeq2SeqDistiller(Seq2SeqDistiller):
    def _set_scorers(self) -> None:
        self.s_scorer = _NoOpScorer()
        self.s_valid_scorer = _NoOpScorer()
        self.s_test_scorer = _NoOpScorer()


def _training_config() -> DictConfig:
    return OmegaConf.create(
        {
            "alpha": 0.5,
            "discriminative_learning": False,
            "learning_rates": [0.1],
            "logging_steps": 10,
            "lr_scheduler": False,
            "optimizer": "sgd",
            "weight_decay": 0.0,
        }
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "t_input_ids": torch.tensor([[1, 2]]),
        "t_attention_mask": torch.tensor([[1, 1]]),
        "t_labels": torch.tensor([[0, 1]]),
        "s_input_ids": torch.tensor([[2, 1]]),
        "s_attention_mask": torch.tensor([[1, 1]]),
        "s_labels": torch.tensor([[0, 1]]),
    }


def test_seq2seq_distillation_trains_student_with_synthetic_batches():
    teacher = _TinySeq2SeqModel()
    student = _TinySeq2SeqModel()
    distiller = _OfflineSeq2SeqDistiller(teacher, student, _training_config())
    initial_weights = student.classifier.weight.detach().clone()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=2,
    )

    trainer.fit(
        distiller, train_dataloaders=DataLoader([_batch(), _batch()], batch_size=None)
    )

    assert trainer.global_step == 2
    assert not torch.equal(student.classifier.weight, initial_weights)
    assert all(parameter.grad is None for parameter in teacher.parameters())
