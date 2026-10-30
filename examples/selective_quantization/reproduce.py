# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.8.0", "transformers==5.16.1", "scikit-learn==1.5.2", "pyarrow==19.0.1", "numpy==2.5.4", "tokenizers==0.23.3", "huggingface-hub==1.33.0", "safetensors==0.8.0"]
# ///
"""Reproduce the frozen UltraBERT quantization study without a Modal account."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import logging
import platform
import random
import time
from pathlib import Path

import compare
import numpy as np
import prepare_data
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from package_checkpoint import sha256, verify_manifest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def rebuild_data(source: Path, output: Path, study: dict) -> list[dict]:
    if sha256(source) != study["source_sha256"]:
        raise ValueError("Pinned source parquet checksum mismatch")
    records, _ = prepare_data.prepare(pq.read_table(source).to_pylist())
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    output.write_bytes(payload.encode("utf-8"))
    if (
        sha256(output) != study["records_sha256"]
        or prepare_data.LABELS != study["labels"]
    ):
        raise ValueError("Rebuilt records/splits differ from the frozen study")
    return records


def select_rows(records: list[dict], study: dict) -> tuple[list[dict], list[dict]]:
    validation = {
        row["text_hash"]: row for row in records if row["split"] == "validation"
    }
    hashes = study["diagnostic_hashes"]
    if len(hashes) != len(set(hashes)) or not set(hashes) <= set(validation):
        raise ValueError("Diagnostic hashes are duplicated or outside validation")
    diagnostic = [validation[key] for key in hashes]
    remainder = [
        row
        for row in records
        if row["split"] == "validation" and row["text_hash"] not in set(hashes)
    ]
    if not diagnostic or not remainder:
        raise ValueError("Expected nonempty diagnostic and remainder subsets")
    return diagnostic, remainder


def verify_release(checkpoint: Path, study: dict) -> dict:
    manifest = verify_manifest(checkpoint, study)
    release = study.get("model_release")
    if (
        release
        and sha256(checkpoint / "release-manifest.json") != release["manifest_sha256"]
    ):
        raise ValueError("Release manifest differs from the pinned publication")
    return manifest


def encode(tokenizer, rows: list[dict]) -> dict[str, torch.Tensor]:
    return dict(
        tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
    )


def payload_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.iterdir()
        if item.is_file()
        and item.suffix
        in {".json", ".py", ".safetensors", ".bin", ".pt", ".txt", ".model"}
        and item.name not in {"release-manifest.json", "README.md", "LICENSE", "NOTICE"}
    )


@torch.inference_mode()
def benchmark(
    model,
    tokenizer,
    rows: list[dict],
    inputs: dict[str, torch.Tensor],
    output: Path,
    *,
    trim: bool,
) -> dict:
    def sample(index):
        length = int(inputs["attention_mask"][index].sum()) if trim else 256
        return {key: value[index : index + 1, :length] for key, value in inputs.items()}

    indices = random.Random(42).sample(range(len(rows)), min(32, len(rows)))
    for index in (indices * 20)[:20]:
        model(**sample(index))
    timings = []
    for repeat in range(3):
        order = indices.copy()
        random.Random(42 + repeat).shuffle(order)
        for index in order:
            prepared = sample(index)
            paths = (
                ("forward", "tokenize_and_forward")
                if (index + repeat) % 2
                else ("tokenize_and_forward", "forward")
            )
            durations = {}
            for name in paths:
                start = time.perf_counter_ns()
                values = (
                    prepared
                    if name == "forward"
                    else dict(
                        tokenizer(
                            rows[index]["text"],
                            truncation=True,
                            max_length=256,
                            padding=False if trim else "max_length",
                            return_tensors="pt",
                        )
                    )
                )
                model(**values)
                durations[name] = (time.perf_counter_ns() - start) / 1e6
            timings.append(
                {"text_hash": rows[index]["text_hash"], "repeat": repeat, **durations}
            )
    output.write_text("".join(json.dumps(row) + "\n" for row in timings))
    return {
        "unique_texts": len(indices),
        "repeats": 3,
        "warmup_calls": 20,
        "samples_per_path": len(timings),
        "batch_size": 1,
        "max_length": 256,
        "padding_length": None if trim else 256,
        "median_input_tokens": float(
            np.median([sample(i)["input_ids"].shape[1] for i in indices])
        ),
        **{
            name
            + "_ms": {
                "p50": float(np.percentile([row[name] for row in timings], 50)),
                "p95": float(np.percentile([row[name] for row in timings], 95)),
            }
            for name in ("forward", "tokenize_and_forward")
        },
    }


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(
    checkpoint: Path, source: Path, output: Path, study: dict, *, timing: bool
) -> dict:
    if output.exists():
        raise ValueError("Output already exists")
    verify_release(checkpoint, study)
    output.mkdir(parents=True)
    records = rebuild_data(source, output / "records.jsonl", study)
    diagnostic, remainder = select_rows(records, study)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.padding_side != "right":
        raise ValueError("The frozen protocol requires right padding")
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        .cpu()
        .eval()
    )
    if [model.config.id2label[i] for i in range(len(study["labels"]))] != study["labels"]:
        raise ValueError("Loaded model label order differs")
    names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    exempt = study["exempt_module"]
    if (
        len(names) != study["expected_linear_modules"]
        or exempt not in names
        or sum(parameter.numel() for parameter in model.parameters())
        != study["expected_parameters"]
    ):
        raise ValueError("Loaded model architecture differs from the measured checkpoint")
    inputs = encode(tokenizer, diagnostic)
    reference = compare.predict(model, inputs)
    reference_metrics = compare.metrics(
        {"batch16_pad256": reference},
        {"batch16_pad256": reference},
        [row["label"] for row in diagnostic],
        len(study["labels"]),
    )
    observed = reference_metrics["batch16_pad256"]["macro_f1"]
    expected = study["recorded_results"]["diagnostic"]["fp32"]["macro_f1"]
    report = {
        "status": "running",
        "scope": study["scope"],
        "diagnostic_hashes": [row["text_hash"] for row in diagnostic],
        "remainder_hashes": [row["text_hash"] for row in remainder],
        "records_sha256": study["records_sha256"],
        "release_manifest_sha256": sha256(checkpoint / "release-manifest.json"),
        "checkpoint_reference_matches": bool(abs(observed - expected) < 1e-8),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": platform.processor() or "unknown",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "scikit-learn",
                "pyarrow",
                "numpy",
                "tokenizers",
                "huggingface-hub",
                "safetensors",
            )
        },
        "engine": torch.backends.quantized.engine,
        "diagnostic": {"fp32": reference_metrics},
        "validation_remainder": {},
    }
    save_json(output / "report.json", report)
    if not report["checkpoint_reference_matches"]:
        raise ValueError(
            "FP32 diagnostic score does not match the recorded checkpoint; inspect report.json before continuing"
        )
    policies = [
        ("int8_all", []),
        ("first_down_only", sorted(names - {exempt})),
        ("int8_selective", [exempt]),
    ]
    for name, exclusions in policies:
        LOGGER.info("Diagnostic: %s", name)
        quantized, selected = compare.convert(model, exclusions)
        logits = compare.predict(quantized, inputs)
        report["diagnostic"][name] = {
            "quantized_modules": selected,
            "quality": compare.metrics(
                {"batch16_pad256": logits},
                {"batch16_pad256": reference},
                [row["label"] for row in diagnostic],
                len(study["labels"]),
            ),
        }
        save_json(output / (name + "-diagnostic-logits.json"), logits.tolist())
        save_json(output / "report.json", report)
        del quantized
        gc.collect()
    LOGGER.info("Diagnostic: rounded FFN weights with FP32 computation")
    quantized, _ = compare.convert(model, [])
    rounded = copy.deepcopy(model)
    with torch.no_grad():
        for name in study["ffn_modules"]:
            rounded.get_submodule(name).weight.copy_(
                quantized.get_submodule(name).weight().dequantize()
            )
    rounded_logits = compare.predict(rounded, inputs)
    report["diagnostic"]["ffn_weight_roundtrip_fp32_compute"] = compare.metrics(
        {"batch16_pad256": rounded_logits},
        {"batch16_pad256": reference},
        [row["label"] for row in diagnostic],
        len(study["labels"]),
    )
    save_json(output / "weight-roundtrip-diagnostic-logits.json", rounded_logits.tolist())
    save_json(output / "report.json", report)
    del rounded, quantized
    gc.collect()
    remainder_inputs = encode(tokenizer, remainder)
    LOGGER.info("FP32: three policies on %d remainder articles", len(remainder))
    fp32_outputs = compare.policy_logits(model, remainder_inputs)
    report["validation_remainder"]["fp32"] = {
        "quality": compare.metrics(
            fp32_outputs,
            fp32_outputs,
            [row["label"] for row in remainder],
            len(study["labels"]),
        ),
        "inference_payload_bytes": payload_bytes(checkpoint),
    }
    if timing:
        report["validation_remainder"]["fp32"]["latency"] = {
            name: benchmark(
                model,
                tokenizer,
                remainder,
                remainder_inputs,
                output / ("fp32-" + name + ".jsonl"),
                trim=trim,
            )
            for name, trim in (("pad256", False), ("actual_length", True))
        }
    save_json(output / "report.json", report)
    quantized, selected = compare.convert(model, [exempt])
    before = compare.predict(quantized, remainder_inputs)
    probes = {key: value[:32] for key, value in remainder_inputs.items()}
    probe_before = compare.policy_logits(quantized, probes)
    artifact = output / "int8_selective"
    compare.save_artifact(
        quantized,
        tokenizer,
        checkpoint,
        artifact,
        {
            "engine": torch.backends.quantized.engine,
            "quantized_modules": selected,
            "excluded_modules": [exempt],
        },
    )
    del quantized, model
    gc.collect()
    restored = compare.load_artifact(artifact, trust_remote_code=True)
    after = compare.predict(restored, remainder_inputs)
    torch.testing.assert_close(after, before)
    probe_after = compare.policy_logits(restored, probes)
    for name in probe_before:
        torch.testing.assert_close(probe_before[name], probe_after[name])
    outputs = compare.policy_logits(restored, remainder_inputs, batch16_logits=after)
    report["validation_remainder"]["int8_selective"] = {
        "quality": compare.metrics(
            outputs,
            fp32_outputs,
            [row["label"] for row in remainder],
            len(study["labels"]),
        ),
        "inference_payload_bytes": payload_bytes(artifact),
        "reload_verified_batch16_rows": len(remainder),
        "reload_verified_batch_one_probes": min(32, len(remainder)),
    }
    if timing:
        report["validation_remainder"]["int8_selective"]["latency"] = {
            name: benchmark(
                restored,
                tokenizer,
                remainder,
                remainder_inputs,
                output / ("int8-" + name + ".jsonl"),
                trim=trim,
            )
            for name, trim in (("pad256", False), ("actual_length", True))
        }
    for variant, values in (("fp32", fp32_outputs), ("int8", outputs)):
        for policy, logits in values.items():
            save_json(output / (variant + "-" + policy + "-logits.json"), logits.tolist())
    report["status"] = "completed"
    save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local packaged release; otherwise use study.json's pinned public release",
    )
    parser.add_argument(
        "--source", type=Path, help="Optional already-downloaded pinned source parquet"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Run all quality/reload checks but omit timing",
    )
    args = parser.parse_args()
    study = json.loads((ROOT / "study.json").read_text())
    if args.output.exists():
        parser.error("Choose a new output directory")
    if "x86" not in torch.backends.quantized.supported_engines:
        parser.error(
            "The study requires an x86 CPU; the generic compare.py demo supports other engines."
        )
    checkpoint = args.checkpoint
    if checkpoint is None:
        release = study.get("model_release")
        if not release:
            parser.error(
                "The measured checkpoint is not published yet. Supply --checkpoint with the local packaged release."
            )
        checkpoint = Path(
            snapshot_download(release["repo_id"], revision=release["revision"])
        )
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if "x86" not in torch.backends.quantized.supported_engines:
        parser.error(
            "The study requires the x86 engine; use an x86 CPU environment. The generic compare.py demo can use other engines."
        )
    torch.backends.quantized.engine = "x86"
    source = args.source or Path(
        hf_hub_download(
            study["dataset"],
            filename=study["parquet_file"],
            repo_type="dataset",
            revision=study["dataset_revision"],
        )
    )
    report = run(checkpoint, source, args.output, study, timing=not args.no_timing)
    LOGGER.info("Completed: %s", args.output / "report.json")
    LOGGER.info(
        "Validation remainder macro-F1: %s",
        {
            name: row["macro_f1"]
            for name, row in report["validation_remainder"]["int8_selective"][
                "quality"
            ].items()
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
