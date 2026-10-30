"""Freeze the five-class newsletter pilot from the pinned source parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

DATASET = "JulesBelveze/tldr_news"
REVISION = "5ee6c613484aa6ac96fac7827a5659409897cc5b"
SOURCE_SHA256 = "50fcbcc76dca3fc6f5057c92f585657855805ef359321488d32f01fe41d0a17d"
# Frozen cross-newsletter taxonomy used by the article study.
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
}
LABELS = sorted(set(SECTION_MAP.values()))


def text_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def prepare(rows: list[dict[str, object]]) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    excluded = Counter()
    for index, row in enumerate(rows):
        section, text = row.get("section"), row.get("text")
        if not isinstance(section, str) or section not in SECTION_MAP:
            excluded["unmapped_section"] += 1
            continue
        if not isinstance(text, str) or not text.strip():
            excluded["empty_text"] += 1
            continue
        groups[text_hash(text)].append(
            {
                "source_row": index,
                "text": text,
                "label": LABELS.index(SECTION_MAP[section]),
            }
        )

    records, conflicts = [], []
    for digest, group in sorted(groups.items()):
        if len({row["label"] for row in group}) > 1:
            conflicts.append({"text_hash": digest, "rows": group})
            excluded["conflicting_label_rows"] += len(group)
            continue
        excluded["duplicate_rows"] += len(group) - 1
        records.append({"text_hash": digest, **group[0]})

    train, heldout = train_test_split(
        records, test_size=0.2, random_state=42, stratify=[r["label"] for r in records]
    )
    validation, test = train_test_split(
        heldout, test_size=0.5, random_state=42, stratify=[r["label"] for r in heldout]
    )
    counts = {}
    for split, subset in (("train", train), ("validation", validation), ("test", test)):
        for row in subset:
            row["split"] = split
        counts[split] = dict(Counter(LABELS[row["label"]] for row in subset))
        if set(counts[split]) != set(LABELS):
            raise ValueError(f"Missing classes in {split}")
    assert len(records) + sum(excluded.values()) == len(rows)
    return records, {
        "source_rows": len(rows),
        "excluded": dict(excluded),
        "class_counts": counts,
        "conflicts": conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Parquet downloaded at REVISION")
    parser.add_argument("output", type=Path, help="New directory; existing paths refused")
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("Source parquet does not match the pinned dataset revision")
    records, audit = prepare(pq.read_table(args.source).to_pylist())
    conflicts = audit.pop("conflicts")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "source_sha256": source_hash,
        "records_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "labels": LABELS,
        "seed": 42,
        "section_map": SECTION_MAP,
        "normalization": "NFKC, casefold, collapse whitespace; used for dedup only",
        **audit,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "records.jsonl").write_text(payload, encoding="utf-8")
    (args.output / "conflicts.json").write_text(
        json.dumps(conflicts, indent=2), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
