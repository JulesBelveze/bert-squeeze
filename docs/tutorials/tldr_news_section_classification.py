"""
Fine-tune ModernBERT on the TLDR-news *section* classification task and compare it
against squeezed variants (distillation, quantization, pruning, early-exiting).

Dataset: https://huggingface.co/datasets/JulesBelveze/tldr_news
    Columns: category, section, title, text, url, newsletter_url. Single `train`
    split (~22k rows). The raw `section` column has ~23 noisy/overlapping strings; we
    merge them into a cleaner 6-class taxonomy (see SECTION_MAP) and classify
    `text` -> `section`. `category` is uniformly "ai" and therefore unusable. The data
    module class-encodes the (remapped) string labels and stratifies the split on them.

What this shows:
    1. Train a ModernBERT teacher with `TrainAssistant("automodel", ...)`.
    2. Distill it into a smaller ModernBERT student while pruning + quantizing.
    3. Include a BERT-based early-exit model (FastBERT) for breadth.
    4. Tabulate params / on-disk size / CPU latency / macro-F1 per variant.

Note on early-exiting: FastBERT/DeeBERT/BERxiT/LayerSkip are their own BERT-based
architectures (not callbacks), so they run on a BERT backbone rather than ModernBERT.
A ModernBERT-native early-exit model would be a separate, larger piece of work.

Run:
    uv run python docs/tutorials/tldr_news_section_classification.py --smoke
    uv run python docs/tutorials/tldr_news_section_classification.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Dict, List

import datasets
import torch
from lightning.pytorch import Trainer
from sklearn.metrics import f1_score
from tabulate import tabulate

from bert_squeeze.assistants import DistilAssistant, TrainAssistant

DATASET_PATH = "JulesBelveze/tldr_news"
TEXT_COL = "text"  # body carries the signal; concat `title` too if you preprocess it
LABEL_COL = "section"
TEACHER_MODEL = "answerdotai/ModernBERT-base"
STUDENT_MODEL = (
    "answerdotai/ModernBERT-base"  # swap for a smaller ModernBERT if available
)
# BERT backbone for FastBERT. Needs a safetensors checkpoint: transformers refuses
# to torch.load .bin files on torch < 2.6 (CVE-2025-32434).
EARLY_EXIT_MODEL = "bert-base-uncased"


# The raw dataset has ~23 `section` strings that are noisy and overlapping (several
# near-synonyms, plus an empty label and ads). We merge them into a cleaner 6-class
# taxonomy; labels absent from this map (e.g. "", "Sponsor", "Miscellaneous") are
# dropped by the data module. Tune this to taste.
SECTION_MAP = {
    "Headlines & Launches": "Headlines & News",
    "Headlines & Trends": "Headlines & News",
    "News & Trends": "Headlines & News",
    "Innovation & Launches": "Headlines & News",
    "Launches & Products": "Headlines & News",
    "Launches & Tools": "Headlines & News",
    "Engineering & Research": "Research & Engineering",
    "Research & Innovation": "Research & Engineering",
    "Deep Dives & Analysis": "Research & Engineering",
    "Deep Dives & Reports": "Research & Engineering",
    "Articles & Tutorials": "Guides & Resources",
    "Guides & Resources": "Guides & Resources",
    "Resources & Tools": "Guides & Resources",
    "Tools & Resources": "Guides & Resources",
    "Opinions & Tutorials": "Guides & Resources",
    "Attacks & Vulnerabilities": "Security",
    "Markets & Business": "Business & Strategy",
    "Strategies & Tactics": "Business & Strategy",
    "Opinions & Advice": "Business & Strategy",
    "Quick Links": "Quick Links",
}


def get_section_labels() -> List[int]:
    """Integer id list for the cleaned taxonomy (the canonical classes of SECTION_MAP)."""
    names = sorted(set(SECTION_MAP.values()))
    print(f"Using {len(names)} cleaned sections: {names}")
    return list(range(len(names)))


FULL_MAX_LEN = 256
SMOKE_MAX_LEN = 32  # tiny seq keeps MPS memory in check (batch 32 x 256 thrashes it)
SMOKE_BATCH = 8
SMOKE_PERCENT = 5  # small but still stratifiable across the cleaned sections


def _data_kwargs(smoke: bool) -> Dict:
    dataset_config = {
        "path": DATASET_PATH,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
        "label_map": SECTION_MAP,  # merge synonyms + drop junk labels
        "stratify_by_column": LABEL_COL,  # works now that labels are class-encoded
    }
    kwargs = {"max_length": FULL_MAX_LEN, "dataset_config": dataset_config}
    if smoke:
        dataset_config["percent"] = SMOKE_PERCENT
        kwargs["max_length"] = SMOKE_MAX_LEN
        kwargs["train_batch_size"] = SMOKE_BATCH
        kwargs["eval_batch_size"] = SMOKE_BATCH
    return kwargs


def macro_f1(model, dataloader) -> float:
    """Macro-F1 over a dataloader, independent of the internal scorer."""
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            if not isinstance(logits, torch.Tensor):  # early-exit models return tuples
                logits = logits[0] if isinstance(logits, tuple) else logits.logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            golds.extend(batch["labels"].cpu().tolist())
    return f1_score(golds, preds, average="macro")


def profile(name: str, model, dataloader) -> Dict:
    """Params, on-disk size (MB), CPU latency (ms/batch) and macro-F1."""
    n_params = sum(p.numel() for p in model.parameters())
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        torch.save(model.state_dict(), fh.name)
        size_mb = os.path.getsize(fh.name) / 1e6
    os.unlink(fh.name)

    batch = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        for _ in range(2):  # warmup
            model.forward(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
        start = time.perf_counter()
        for _ in range(5):
            model.forward(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
        latency_ms = (time.perf_counter() - start) / 5 * 1000

    return {
        "variant": name,
        "params (M)": round(n_params / 1e6, 1),
        "size (MB)": round(size_mb, 1),
        "latency (ms/batch)": round(latency_ms, 1),
        "macro-F1": round(macro_f1(model, dataloader), 3),
    }


def _distill_data_kwargs(smoke: bool) -> Dict:
    """Apply the same max_length / batch / dataset overrides to both distil modules."""
    dk = _data_kwargs(smoke)
    module = {k: v for k, v in dk.items() if k != "dataset_config"}
    return {
        "teacher_module": {**module, "dataset_config": dict(dk["dataset_config"])},
        "student_module": {**module, "dataset_config": dict(dk["dataset_config"])},
    }


def _general(labels: List[int]) -> Dict:
    return {"labels": labels, "num_labels": len(labels)}


def _make_trainer(smoke: bool, max_steps: int, epochs: int, callbacks=None) -> Trainer:
    if smoke:
        return Trainer(max_steps=max_steps, callbacks=callbacks)
    return Trainer(max_epochs=epochs, callbacks=callbacks)


def train_teacher(labels: List[int], smoke: bool, max_steps: int, epochs: int):
    assistant = TrainAssistant(
        "automodel",
        general_kwargs=_general(labels),
        train_kwargs={"objective": "ce", "num_epochs": epochs},
        model_kwargs={"pretrained_model": TEACHER_MODEL},
        data_kwargs=_data_kwargs(smoke),
    )
    _make_trainer(smoke, max_steps, epochs).fit(
        assistant.model,
        train_dataloaders=assistant.data.train_dataloader(),
        val_dataloaders=assistant.data.val_dataloader(),
    )
    return assistant


def distill_student(labels: List[int], smoke: bool, max_steps: int, epochs: int):
    """Distill ModernBERT -> ModernBERT (labelled data) while pruning + quantizing.

    For a real run, point the teacher at your trained checkpoint via
    ``teacher_kwargs={"checkpoint_path": "<lightning ckpt>"}`` — here it starts from
    the pretrained weights so the tutorial is self-contained.
    """
    assistant = DistilAssistant(
        "distil",
        general_kwargs=_general(labels),
        teacher_kwargs={
            "pretrained_model_name_or_path": TEACHER_MODEL,
            "num_labels": len(labels),
        },
        student_kwargs={
            "pretrained_model_name_or_path": STUDENT_MODEL,
            "num_labels": len(labels),
        },
        data_kwargs=_distill_data_kwargs(smoke),
        callbacks=[
            {
                "_target_": "bert_squeeze.utils.callbacks.pruning.ThresholdBasedPruning",
                "threshold": 0.2,
                "start_pruning_epoch": -1,
            },
            {"_target_": "bert_squeeze.utils.callbacks.quantization.DynamicQuantization"},
        ],
    )
    _make_trainer(smoke, max_steps, epochs, assistant.callbacks).fit(
        assistant.model,
        train_dataloaders=assistant.data.train_dataloader(),
        val_dataloaders=assistant.data.test_dataloader(),
    )
    return assistant


def train_early_exit(labels: List[int], smoke: bool, max_steps: int, epochs: int):
    """FastBERT (BERT-based early-exit) for the comparison breadth."""
    assistant = TrainAssistant(
        "fastbert",
        general_kwargs=_general(labels),
        train_kwargs={"objective": "ce", "num_epochs": epochs},
        model_kwargs={"pretrained_model": EARLY_EXIT_MODEL},
        data_kwargs={"tokenizer_name": EARLY_EXIT_MODEL, **_data_kwargs(smoke)},
    )
    _make_trainer(smoke, max_steps, epochs, assistant.callbacks).fit(
        assistant.model,
        train_dataloaders=assistant.data.train_dataloader(),
        val_dataloaders=assistant.data.test_dataloader(),
    )
    return assistant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (5%% of data, few steps) to validate the pipeline end to end.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    max_steps = 5 if args.smoke else -1

    labels = get_section_labels()

    teacher = train_teacher(labels, args.smoke, max_steps, args.epochs)
    student = distill_student(labels, args.smoke, max_steps, args.epochs)
    early_exit = train_early_exit(labels, args.smoke, max_steps, args.epochs)

    # The distillation loader yields t_/s_-prefixed columns; profile the student on the
    # teacher's plain loader instead (both share the ModernBERT tokenizer).
    rows = [
        profile("modernbert-teacher", teacher.model, teacher.data.test_dataloader()),
        profile(
            "modernbert-student", student.model.student, teacher.data.test_dataloader()
        ),
        profile("bert-fastbert", early_exit.model, early_exit.data.test_dataloader()),
    ]
    print("\n" + tabulate(rows, headers="keys", tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
