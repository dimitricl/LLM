#!/usr/bin/env python3
"""Build the anti-leak index from SFT val/test and future evaluation data.

SFT val/test are identified exactly like ``scripts/../src/tinyllm/sft_data.py``
does it: normalise the messages, then apply ``stable_split`` with the same
``val_frac``/``test_frac``.  The resulting conversations, plus any file found
under ``data/eval``, are written to a compact JSONL index consumed by the
filtering script.

The script only reads source files.  Its output path must be new (or
``--force`` must be explicit).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tinyllm.pretrain_v8 import json_dump, leak_texts
from tinyllm.sft_data import normalize_messages, stable_split


def iter_sft_val_test(
    path: Path,
    val_frac: float,
    test_frac: float,
    max_docs: int | None,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {
        "examined": 0,
        "valid": 0,
        "val": 0,
        "test": 0,
        "rejected": 0,
        "rejected_reasons": Counter(),
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_docs is not None and stats["examined"] >= max_docs:
                break
            if not line.strip():
                continue
            stats["examined"] += 1
            try:
                record = json.loads(line)
                messages = normalize_messages(record.get("messages"))
            except Exception as exc:
                stats["rejected"] += 1
                stats["rejected_reasons"][str(exc)] += 1
                continue
            stats["valid"] += 1
            split = stable_split(messages, val_frac, test_frac)
            if split not in ("val", "test"):
                continue
            stats[split] += 1
            records.append(
                {
                    "kind": "sft_val_test",
                    "split": split,
                    "messages": messages,
                }
            )
    stats["rejected_reasons"] = dict(stats["rejected_reasons"].most_common())
    return records, stats


def iter_eval_dir(path: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {"files": 0, "records": 0, "skipped_files": []}
    if not path.is_dir():
        return records, stats

    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in (".jsonl", ".json", ".txt"):
            stats["skipped_files"].append(str(file_path))
            continue
        stats["files"] += 1
        if suffix == ".txt":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append({"kind": "eval", "text": line})
                        stats["records"] += 1
            continue
        if suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict):
                        for text in leak_texts(record):
                            records.append({"kind": "eval", "text": text})
                            stats["records"] += 1
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                for text in leak_texts(item):
                    records.append({"kind": "eval", "text": text})
                    stats["records"] += 1
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage B leak index.")
    parser.add_argument(
        "--sft-file",
        default="data/sft/downloaded/raw/combined_v7.jsonl",
    )
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument(
        "--output",
        default="data/v8/pretrain/leak/leak_index.jsonl",
    )
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--test-frac", type=float, default=0.02)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(
            f"{output} already exists; pass --force to replace it explicitly."
        )
    if not Path(args.sft_file).is_file():
        raise SystemExit(f"SFT file not found: {args.sft_file}")

    sft_records, sft_stats = iter_sft_val_test(
        Path(args.sft_file),
        args.val_frac,
        args.test_frac,
        args.max_docs,
    )
    eval_records, eval_stats = iter_eval_dir(Path(args.eval_dir))
    records = sft_records + eval_records

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    characters = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            for text in leak_texts(record):
                characters += len(text)
    temporary.replace(output)

    summary = {
        "output": str(output),
        "sft_file": args.sft_file,
        "eval_dir": args.eval_dir,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "sft": sft_stats,
        "eval": eval_stats,
        "records_written": len(records),
        "characters": characters,
        "note": (
            "data/eval does not exist yet: anti-leak against the Stage E "
            "evaluation set must be checked again when it is created."
            if not Path(args.eval_dir).is_dir()
            else "evaluation files included"
        ),
    }
    json_dump(output.with_name("leak_index_stats.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not records:
        print(
            "WARNING: the leak index is empty. Check --max-docs or the SFT "
            "split fractions.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
