#!/usr/bin/env python3
"""Tokenize filtered documents into flat uint16 binary streams.

The script reads the filtered JSONL output from ``filter_pretrain_v8.py``,
tokenizes each document with the v8 tokenizer, appends EOS id (0) at the end,
and writes a flat uint16 stream to ``.bin`` files for train/val/test.

Document split is already decided by the filter script: the ``split`` field
in each record is one of ``train``, ``val`` or ``test``.  This script respects
that split and never modifies it.

EOS handling
------------
The v8 tokenizer never emits id 0 from ``encode(text)``.  This script
explicitly appends the EOS id (0) to the end of each document.  The binary
writer validates that:
* Each document ends with exactly one EOS id
* No EOS id appears inside a document
* The number of EOS tokens equals the number of documents

Context length
--------------
The documents are never truncated.  The full document tokens plus the EOS id
are written to the binary stream.  The loader (``TokenDataset``) slides a
context window over the stream.  Documents longer than the context are
split across multiple training windows.

Output format
-------------
Each split (train/val/test) gets a flat ``.bin`` file with uint16 tokens:
    [doc1_token1, doc1_token2, ..., doc1_tokenN, EOS,
     doc2_token1, doc2_token2, ..., doc2_tokenM, EOS,
     ...]

A ``tokenize_stats.json`` file reports:
* Documents and tokens per split
* Number of EOS tokens (must equal document count)
* Mean tokens per document
* Percentage of documents longer than context
* Estimated training time at 7,000 tokens/s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from tinyllm.pretrain_v8 import (
    EOS_ID,
    FlatTokenWriter,
    CONTEXT_LEN,
    estimated_seconds,
    json_dump,
    read_jsonl,
)
from tinyllm.tokenizer import TokenizerWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize filtered pre-train corpus to binary."
    )
    parser.add_argument(
        "--input-dir",
        default="data/v8/pretrain/filtered",
    )
    parser.add_argument(
        "--output-dir",
        default="data/v8/pretrain/tokenized",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/v8/tokenizer.json",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Only process these sources (default: all).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum total documents (for smoke tests).",
    )
    parser.add_argument(
        "--tokens-per-second",
        type=float,
        default=7000.0,
        help="Training throughput for time estimation.",
    )
    args = parser.parse_args()

    if args.max_docs is not None and args.max_docs < 1:
        parser.error("--max-docs must be positive.")
    if args.tokens_per_second <= 0:
        parser.error("--tokens-per-second must be positive.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")

    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists. Choose a new output directory; "
            "this script never overwrites tokenized data."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = TokenizerWrapper(tokenizer_path)
    if tokenizer.eos_id != EOS_ID:
        raise SystemExit(
            f"Tokenizer eos_id ({tokenizer.eos_id}) != expected EOS_ID ({EOS_ID})."
        )

    source_files = sorted(input_dir.glob("*.jsonl"))
    if args.sources:
        source_files = [f for f in source_files if f.stem in args.sources]

    if not source_files:
        raise SystemExit(f"No source JSONL files found in {input_dir}")

    writers = {
        "train": FlatTokenWriter(output_dir / "train.bin"),
        "val": FlatTokenWriter(output_dir / "val.bin"),
        "test": FlatTokenWriter(output_dir / "test.bin"),
    }

    stats = {
        "split": Counter(),
        "eos_in_encoded": Counter(),
        "rejected": Counter(),
    }
    total_docs = 0
    total_tokens = 0
    started = time.time()

    try:
        for source_file in source_files:
            source_name = source_file.stem
            print(f"=== {source_name} ===", flush=True)
            for record in read_jsonl(source_file):
                if args.max_docs is not None and total_docs >= args.max_docs:
                    break

                split = record.get("split")
                if split not in writers:
                    stats["rejected"][f"invalid_split_{split}"] += 1
                    continue

                text = record.get("text")
                if not isinstance(text, str):
                    stats["rejected"]["missing_text"] += 1
                    continue

                ids = tokenizer.encode(text)
                if EOS_ID in ids:
                    stats["eos_in_encoded"][source_name] += 1
                    stats["rejected"][f"eos_in_encoded_{source_name}"] += 1
                    continue

                ids.append(EOS_ID)
                writers[split].add_document(ids, context_len=CONTEXT_LEN)

                stats["split"][split] += 1
                total_docs += 1
                total_tokens += len(ids)

                if total_docs % 1000 == 0:
                    elapsed = max(1e-9, time.time() - started)
                    print(
                        f"[tokenize] {total_docs:,} docs / "
                        f"{total_tokens:,} tokens / "
                        f"{total_docs / elapsed:.1f} docs/s",
                        flush=True,
                    )
    finally:
        for writer in writers.values():
            writer.close()

    split_summaries = {}
    for split, writer in writers.items():
        summary = writer.summary(context_len=CONTEXT_LEN)
        summary["estimated_hours"] = estimated_seconds(
            summary["tokens"], args.tokens_per_second
        ) / 3600.0
        split_summaries[split] = summary

    totals = {
        "documents": total_docs,
        "tokens": total_tokens,
        "eos_tokens": sum(s["eos_tokens"] for s in split_summaries.values()),
        "estimated_hours": estimated_seconds(total_tokens, args.tokens_per_second)
        / 3600.0,
    }

    summary = {
        "format_version": 1,
        "stage": "tokenize",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "context_len": CONTEXT_LEN,
        "tokens_per_second": args.tokens_per_second,
        "sources_processed": [f.stem for f in source_files],
        "stats": {
            "split": dict(stats["split"]),
            "eos_in_encoded": dict(stats["eos_in_encoded"]),
            "rejected": dict(stats["rejected"]),
        },
        "splits": split_summaries,
        "totals": totals,
        "elapsed_s": round(time.time() - started, 3),
    }

    json_dump(output_dir / "tokenize_stats.json", summary)
    print(json.dumps(totals, indent=2, ensure_ascii=False))

    if stats["eos_in_encoded"]:
        print(
            f"WARNING: found EOS id {EOS_ID} in encoded text for "
            f"{sum(stats['eos_in_encoded'].values())} documents. "
            "These were rejected.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
