# /// script
# requires-python = ">=3.10,<3.13"
# ///
"""Prepare a reviewed model release from the downloaded selected checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = (
    "config.json",
    "configuration_ultrabert.py",
    "modeling_ultrabert.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(checkpoint: Path, study: dict) -> dict:
    manifest = json.loads((checkpoint / "release-manifest.json").read_text())
    if not set(FILES) <= set(manifest["files"]):
        raise ValueError("Release manifest omits required model files")
    for name, expected in study.get("verified_checkpoint_files", {}).items():
        if manifest["files"].get(name) != expected:
            raise ValueError(
                f"Release differs from the numerically verified checkpoint: {name}"
            )
    if (
        manifest["records_sha256"] != study["records_sha256"]
        or manifest["labels"] != study["labels"]
        or manifest["base_revision"] != study["base_revision"]
    ):
        raise ValueError("Checkpoint release provenance differs from study")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint file checksum mismatch: {name}")
    return manifest


def package(checkpoint: Path, output: Path, study: dict) -> dict:
    if output.exists():
        raise ValueError("Release output already exists")
    for name in FILES:
        if not (checkpoint / name).is_file() or (checkpoint / name).is_symlink():
            raise ValueError(f"Missing regular checkpoint file: {name}")
    for name, expected in study.get("verified_checkpoint_files", {}).items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(
                f"Source differs from the numerically verified checkpoint: {name}"
            )
    config = json.loads((checkpoint / "config.json").read_text())
    if (
        config.get("architectures") != [study["expected_architecture"]]
        or [config.get("id2label", {}).get(str(i)) for i in range(5)] != study["labels"]
    ):
        raise ValueError("Checkpoint architecture or label order differs from the study")
    expected_bytes = study["recorded_results"]["validation_remainder"]["ultrabert-fp32"][
        "weight_bytes"
    ]
    if (checkpoint / "model.safetensors").stat().st_size != expected_bytes:
        raise ValueError(
            "Checkpoint weight-file size differs from the recorded FP32 artifact"
        )
    output.mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(checkpoint / name, output / name)
    for name in ("LICENSE", "NOTICE"):
        if (checkpoint / name).is_file():
            shutil.copyfile(checkpoint / name, output / name)
    if not (output / "LICENSE").exists():
        shutil.copyfile(ROOT / "LICENSE.checkpoint", output / "LICENSE")
    if not (output / "NOTICE").exists():
        (output / "NOTICE").write_text(
            "Fine-tuned from dmis-lab/UltraBERT-150M, revision "
            + study["base_revision"]
            + ".\nUpstream model: https://huggingface.co/dmis-lab/UltraBERT-150M\nModel code uses Hugging Face Transformers. Original source files are retained.\n"
        )
    shutil.copyfile(ROOT / "MODEL_CARD.md", output / "README.md")
    manifest = {
        "source_run": "ultrabert-full-db1117fb7a32",
        "base_model": study["base_model"],
        "base_revision": study["base_revision"],
        "records_sha256": study["records_sha256"],
        "labels": study["labels"],
        "training": study["training_protocol"],
        "verification": "Structural checks and hashes only; run reproduce.py for numerical verification",
        "files": {
            path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def publish(checkpoint: Path, repo_id: str, study_path: Path) -> str:
    from huggingface_hub import HfApi, snapshot_download

    study = json.loads(study_path.read_text())
    manifest = verify_manifest(checkpoint, study)
    local_manifest_hash = sha256(checkpoint / "release-manifest.json")
    api = HfApi()
    # Refuse an existing repository so this command cannot overwrite a prior release.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=checkpoint,
        allow_patterns=list(manifest["files"]) + ["release-manifest.json"],
        commit_message="Release verified UltraBERT quantization study checkpoint",
    )
    downloaded = Path(
        snapshot_download(repo_id=repo_id, revision=commit.oid, repo_type="model")
    )
    verify_manifest(downloaded, study)
    if sha256(downloaded / "release-manifest.json") != local_manifest_hash:
        raise ValueError("Published manifest does not match the verified local release")
    study["model_release"] = {
        "repo_id": repo_id,
        "revision": commit.oid,
        "manifest_sha256": local_manifest_hash,
        "download_verified": True,
        "download_verification_scope": "Release file hashes matched; inference was not rerun after upload",
    }
    study_path.write_text(json.dumps(study, indent=2) + "\n")
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--pin-repo", help="Register an already-published release")
    parser.add_argument("--revision", help="Immutable Hub commit for --pin-repo")
    parser.add_argument(
        "--publish-repo",
        help="Create a NEW public repository, upload, verify downloaded bytes, and pin",
    )
    args = parser.parse_args()
    study = json.loads((ROOT / "study.json").read_text())
    if args.publish_repo:
        if args.output is not None or args.pin_repo or args.revision:
            parser.error(
                "Use --publish-repo without an output path or manual pin options"
            )
        revision = publish(args.checkpoint, args.publish_repo, ROOT / "study.json")
        print(
            f"Published and verified https://huggingface.co/{args.publish_repo}/tree/{revision}"
        )
        print("Pinned study.json locally. Public bytes verified; no inference was rerun.")
        return
    if args.pin_repo:
        if (
            args.output is not None
            or args.revision is None
            or re.fullmatch(r"[0-9a-f]{40}", args.revision) is None
        ):
            parser.error(
                "Use a packaged checkpoint, --pin-repo, and a full --revision without an output path"
            )
        verify_manifest(args.checkpoint, study)
        study["model_release"] = {
            "repo_id": args.pin_repo,
            "revision": args.revision,
            "manifest_sha256": sha256(args.checkpoint / "release-manifest.json"),
        }
        (ROOT / "study.json").write_text(json.dumps(study, indent=2) + "\n")
        print(
            "Registered pinned release locally; no upload or remote validation performed."
        )
        return
    if args.output is None or args.revision:
        parser.error("Supply an output directory to package a checkpoint")
    manifest = package(args.checkpoint, args.output, study)
    print(
        f"Prepared {len(manifest['files'])} files at {args.output}; no upload performed."
    )


if __name__ == "__main__":
    main()
