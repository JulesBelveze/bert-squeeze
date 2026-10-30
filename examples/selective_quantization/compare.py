# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["torch==2.8.0", "transformers==5.16.1", "scikit-learn==1.5.2"]
# ///
"""Compare full and selective CPU INT8 conversion on your labelled examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def read_rows(path: Path, label_count: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation JSONL must contain at least one row")
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
        ):
            raise ValueError("Each row needs a nonempty text string")
        if type(row.get("label")) is not int or not 0 <= row["label"] < label_count:
            raise ValueError("Labels must be integer IDs matching the checkpoint")
    return rows


def convert(model, excluded: list[str]):
    names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    if not set(excluded) <= names or not names - set(excluded):
        raise ValueError(
            f"Unknown exclusions or empty selection: {excluded}; available Linear modules: {sorted(names)}"
        )
    selected = sorted(names - set(excluded))
    converted = torch.ao.quantization.quantize_dynamic(
        model, set(selected), dtype=torch.qint8, inplace=False
    ).eval()
    actual = sorted(
        name
        for name, module in converted.named_modules()
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
    )
    if actual != selected:
        raise ValueError("Actual conversion differs from requested modules")
    return converted, selected


def save_artifact(model, tokenizer, source: Path, output: Path, spec: dict) -> None:
    output.mkdir()
    model.config.save_pretrained(output)
    tokenizer.save_pretrained(output)
    if source.is_dir():
        for code in source.glob("*.py"):
            shutil.copyfile(code, output / code.name)
    torch.save(model.state_dict(), output / "quantized.pt")
    (output / "quantization.json").write_text(json.dumps(spec, indent=2) + "\n")


def load_artifact(path: Path, *, trust_remote_code: bool = False):
    spec = json.loads((path / "quantization.json").read_text())
    if spec["engine"] not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Unavailable quantization engine: {spec['engine']}")
    torch.backends.quantized.engine = spec["engine"]
    config = AutoConfig.from_pretrained(
        path, trust_remote_code=trust_remote_code, local_files_only=True
    )
    model = (
        AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=trust_remote_code
        )
        .cpu()
        .eval()
    )
    excluded = spec["excluded_modules"]
    model, selected = convert(model, excluded)
    if selected != spec["quantized_modules"]:
        raise ValueError("Artifact module selection differs from reconstructed model")
    model.load_state_dict(
        torch.load(path / "quantized.pt", map_location="cpu", weights_only=True)
    )
    return model


@torch.inference_mode()
def predict(
    model, inputs: dict[str, torch.Tensor], *, batch_size: int = 16, trim: bool = False
) -> torch.Tensor:
    if batch_size < 1 or (trim and batch_size != 1):
        raise ValueError("Actual-length evaluation requires batch one")
    parts = []
    for start in range(0, inputs["input_ids"].shape[0], batch_size):
        batch = {key: value[start : start + batch_size] for key, value in inputs.items()}
        if trim:
            length = int(batch["attention_mask"].sum())
            batch = {key: value[:, :length] for key, value in batch.items()}
        logits = model(**batch).logits
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits")
        parts.append(logits)
    return torch.cat(parts)


def policy_logits(
    model, inputs: dict[str, torch.Tensor], *, batch16_logits=None
) -> dict[str, torch.Tensor]:
    return {
        "batch16_pad256": (
            predict(model, inputs) if batch16_logits is None else batch16_logits
        ),
        "batch1_pad256": predict(model, inputs, batch_size=1),
        "batch1_actual_length": predict(model, inputs, batch_size=1, trim=True),
    }


def metrics(
    outputs: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    gold: list[int],
    label_count: int,
) -> dict:
    reports = {}
    for name, values in outputs.items():
        predicted = values.argmax(-1)
        reports[name] = {
            "accuracy": accuracy_score(gold, predicted.tolist()),
            "macro_f1": f1_score(
                gold,
                predicted.tolist(),
                labels=list(range(label_count)),
                average="macro",
                zero_division=0,
            ),
            "changed_from_fp32_same_policy": int(
                (predicted != reference[name].argmax(-1)).sum()
            ),
            "changed_from_own_batch16_pad256": int(
                (predicted != outputs["batch16_pad256"].argmax(-1)).sum()
            ),
            "predictions": predicted.tolist(),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Local HF checkpoint or native Hub model ID")
    parser.add_argument("--revision", help="Required immutable commit for a Hub model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exact Linear module name; repeat as needed",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom code from a trusted local checkpoint",
    )
    args = parser.parse_args()
    source = Path(args.checkpoint)
    if not source.is_dir() and (
        args.revision is None or re.fullmatch(r"[0-9a-f]{40}", args.revision) is None
    ):
        parser.error("Hub checkpoints require --revision with a full 40-character commit")
    if args.trust_remote_code and not source.is_dir():
        parser.error("This example supports custom code only from a local checkpoint")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    engine = next(
        (
            name
            for name in ("x86", "fbgemm", "qnnpack")
            if name in torch.backends.quantized.supported_engines
        ),
        None,
    )
    if engine is None:
        raise RuntimeError("No supported CPU dynamic INT8 engine")
    torch.backends.quantized.engine = engine
    options = {"revision": args.revision, "trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **options)
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.checkpoint, **options)
        .cpu()
        .eval()
    )
    if tokenizer.padding_side != "right":
        raise ValueError("This example expects right-padded tokenization")
    if model.config.problem_type not in (None, "single_label_classification"):
        raise ValueError("This example supports single-label classification")
    rows = read_rows(args.data, model.config.num_labels)
    # Validate the exemption before writing files or running inference.
    names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    if not set(args.exclude) <= names or not names - set(args.exclude):
        raise ValueError(f"Invalid exclusions; available Linear modules: {sorted(names)}")
    inputs = dict(
        tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
    )
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output / "fp32")
    tokenizer.save_pretrained(args.output / "fp32")
    reference = policy_logits(model, inputs)
    report = {
        "scope": "Comparison on supplied examples; no automatic module search or generalization claim",
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(rows),
        "label_counts": {
            str(i): sum(row["label"] == i for row in rows)
            for i in range(model.config.num_labels)
        },
        "labels": model.config.id2label,
        "engine": engine,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "max_length": 256,
        "variants": {
            "fp32": {
                "quality": metrics(
                    reference,
                    reference,
                    [row["label"] for row in rows],
                    model.config.num_labels,
                )
            }
        },
    }
    policies = [("int8_all", [])]
    if args.exclude:
        policies.append(("int8_selective", args.exclude))
    for name, exclusions in policies:
        quantized, selected = convert(model, exclusions)
        expected = policy_logits(quantized, inputs)
        artifact = args.output / name
        save_artifact(
            quantized,
            tokenizer,
            source,
            artifact,
            {
                "engine": engine,
                "quantized_modules": selected,
                "excluded_modules": exclusions,
            },
        )
        restored = load_artifact(artifact, trust_remote_code=args.trust_remote_code)
        actual = policy_logits(restored, inputs)
        for policy in expected:
            torch.testing.assert_close(actual[policy], expected[policy])
        report["variants"][name] = {
            "quantized_linear_modules": len(selected),
            "excluded_modules": exclusions,
            "reload_verified_rows_per_policy": len(rows),
            "quality": metrics(
                actual, reference, [row["label"] for row in rows], model.config.num_labels
            ),
        }
        del restored, quantized
    for name, variant in report["variants"].items():
        variant["deployment_bytes"] = sum(
            path.stat().st_size
            for path in (args.output / name).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Saved comparison and reloaded artifacts to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
