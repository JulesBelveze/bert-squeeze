from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/selective_quantization"


@pytest.fixture
def study_module(tmp_path, monkeypatch):
    isolated = tmp_path / "checkout/examples/selective_quantization"
    shutil.copytree(EXAMPLE, isolated, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.syspath_prepend(str(isolated))
    for name in ("compare", "prepare_data", "package_checkpoint"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "isolated_study", isolated / "reproduce.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isolated_study_rebuilds_data_and_runs_saved_artifacts(tmp_path, study_module):
    m = study_module
    torch.set_num_threads(1)
    source = []
    sections = [
        "Markets & Business",
        "Guides & Resources",
        "News & Trends",
        "Engineering & Research",
        "Attacks & Vulnerabilities",
    ]
    for label, section in enumerate(sections):
        for index in range(10):
            source.append({"section": section, "text": f"article {label} number {index}"})
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(source), parquet)
    records, audit = m.prepare_data.prepare(source)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    diagnostic = next(
        row for row in records if row["split"] == "validation" and row["label"] == 0
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    labels = m.prepare_data.LABELS
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=5,
            id2label=dict(enumerate(labels)),
            label2id={name: i for i, name in enumerate(labels)},
        )
    )
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[0] = 1
    model.save_pretrained(checkpoint)
    backend = Tokenizer(
        WordPiece(
            vocab={
                word: i
                for i, word in enumerate(
                    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "article", "number", "a", "b"]
                )
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(checkpoint)
    for name in ("configuration_ultrabert.py", "modeling_ultrabert.py"):
        (checkpoint / name).write_text(
            "# Native-model test fixture; no custom code executed.\n"
        )
    names = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    study = {
        "source_sha256": m.sha256(parquet),
        "records_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "labels": labels,
        "diagnostic_hashes": [diagnostic["text_hash"]],
        "exempt_module": "bert.encoder.layer.0.output.dense",
        "expected_linear_modules": len(names),
        "ffn_modules": names,
        "expected_parameters": sum(p.numel() for p in model.parameters()),
        "base_revision": "fixture",
        "model_release": None,
        "scope": "Offline fixture, not study-quality evidence",
        "recorded_results": {"diagnostic": {"fp32": {"macro_f1": 0.2}}},
    }
    manifest = {
        "records_sha256": study["records_sha256"],
        "labels": labels,
        "base_revision": "fixture",
        "files": {path.name: m.sha256(path) for path in checkpoint.iterdir()},
    }
    (checkpoint / "release-manifest.json").write_text(json.dumps(manifest))
    m.torch.backends.quantized.engine = next(
        name
        for name in ("x86", "fbgemm", "qnnpack")
        if name in torch.backends.quantized.supported_engines
    )
    output = tmp_path / "run"
    report = m.run(checkpoint, parquet, output, study, timing=False)
    assert report["status"] == "completed"
    assert report["checkpoint_reference_matches"]
    assert len(report["diagnostic_hashes"]) == 1
    assert len(report["remainder_hashes"]) == 4
    assert (
        report["validation_remainder"]["int8_selective"]["reload_verified_batch16_rows"]
        == 4
    )
    assert (output / "int8_selective/quantized.pt").exists()
    study["verified_checkpoint_files"] = {
        "model.safetensors": m.sha256(checkpoint / "model.safetensors")
    }
    m.verify_release(checkpoint, study)
    altered = json.loads((checkpoint / "release-manifest.json").read_text())
    altered["files"]["model.safetensors"] = "0" * 64
    (checkpoint / "release-manifest.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="numerically verified"):
        m.verify_release(checkpoint, study)
    (checkpoint / "release-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="already exists"):
        m.run(checkpoint, parquet, output, study, timing=False)
    (checkpoint / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        m.verify_release(checkpoint, study)


def test_packager_rejects_wrong_labels_and_omits_unlisted_files(
    tmp_path, study_module, monkeypatch
):
    import package_checkpoint as packager

    source = tmp_path / "source"
    source.mkdir()
    labels = ["a", "b", "c", "d", "e"]
    for name in packager.FILES:
        (source / name).write_bytes(b"fixture")
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["UltrabertForSequenceClassification"],
                "id2label": dict(enumerate(labels)),
            }
        )
    )
    (source / "secret.env").write_text("must not be packaged")
    study = {
        "expected_architecture": "UltrabertForSequenceClassification",
        "labels": labels,
        "base_model": "parent",
        "base_revision": "revision",
        "records_sha256": "frozen",
        "training_protocol": {},
        "recorded_results": {
            "validation_remainder": {"ultrabert-fp32": {"weight_bytes": 7}}
        },
    }
    output = tmp_path / "release"
    manifest = packager.package(source, output, study)
    assert "model.safetensors" in manifest["files"]
    assert not (output / "secret.env").exists()
    assert (output / "README.md").exists() and (output / "LICENSE").exists()
    (packager.ROOT / "study.json").write_text(json.dumps(study))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_checkpoint.py",
            str(output),
            "--pin-repo",
            "owner/fixture",
            "--revision",
            "a" * 40,
        ],
    )
    packager.main()
    registered = json.loads((packager.ROOT / "study.json").read_text())["model_release"]
    assert registered["revision"] == "a" * 40
    assert registered["manifest_sha256"] == packager.sha256(
        output / "release-manifest.json"
    )
    with pytest.raises(ValueError, match="already exists"):
        packager.package(source, output, study)
    study["labels"] = list(reversed(labels))
    with pytest.raises(ValueError, match="label order"):
        packager.package(source, tmp_path / "wrong", study)


def test_pending_public_release_does_not_attempt_download(
    tmp_path, study_module, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "--output", str(tmp_path / "run")])
    monkeypatch.setattr(
        study_module,
        "snapshot_download",
        lambda *args, **kwargs: pytest.fail("Unexpected model download"),
    )
    monkeypatch.setattr(
        study_module,
        "hf_hub_download",
        lambda *args, **kwargs: pytest.fail("Unexpected data download"),
    )
    with pytest.raises(SystemExit):
        study_module.main()


def test_publisher_refuses_overwrite_and_pins_only_after_verified_download(
    tmp_path, monkeypatch
):
    import importlib.util
    from types import SimpleNamespace

    path = EXAMPLE / "package_checkpoint.py"
    spec = importlib.util.spec_from_file_location("publisher_fixture", path)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    files = {}
    for name in publisher.FILES:
        (checkpoint / name).write_bytes(b"fixture")
        files[name] = publisher.sha256(checkpoint / name)
    manifest = {
        "records_sha256": "records",
        "labels": ["label"],
        "base_revision": "base",
        "files": files,
    }
    (checkpoint / "release-manifest.json").write_text(json.dumps(manifest))
    study_path = tmp_path / "study.json"
    study_path.write_text(
        json.dumps(
            {
                "records_sha256": "records",
                "labels": ["label"],
                "base_revision": "base",
                "model_release": None,
            }
        )
    )
    calls = []

    class Api:
        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(oid="a" * 40)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(checkpoint)
    )
    assert publisher.publish(checkpoint, "owner/model", study_path) == "a" * 40
    assert calls[0][1]["exist_ok"] is False and calls[0][1]["private"] is False
    assert set(calls[1][1]["allow_patterns"]) == set(files) | {"release-manifest.json"}
    assert json.loads(study_path.read_text())["model_release"]["revision"] == "a" * 40
    calls.clear()
    (checkpoint / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        publisher.publish(checkpoint, "owner/other", study_path)
    assert not calls
