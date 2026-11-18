#!/usr/bin/env python3
"""Filter Stage B raw JSONL into clean, deduplicated, leak-free documents.

Every examined document is tokenized once so the report gives *exact* token
counts before and after each filter.  A document that fails a length filter is
rejected whole: it is never truncated.  Exact SHA1 dedup and MinHash near-dup
dedup run across all sources.  The leak index removes exact, near-duplicate
and n-gram-containment matches against SFT val/test and ``data/eval``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

from tinyllm.pretrain_v8 import (
    SOURCES,
    LeakIndex,
    SourceSpec,
    VAL_FRAC,
    TEST_FRAC,
    alpha_ratio,
    band_keys,
    counter_dict,
    jaccard_estimate,
    json_dump,
    language_ok,
    make_minhash_params,
    minhash_signature,
    normalize_text,
    read_jsonl,
    sample_words,
    sha1_digest,
    shingle_hashes_from_words,
    split_for_text,
    word_tokens,
)
from tinyllm.tokenizer import TokenizerWrapper

FILTER_NAMES = (
    "length",
    "language",
    "alphabet",
    "repetition",
    "exact_duplicate",
    "near_duplicate",
    "leak",
)


def parse_sources(values: list[str] | None) -> list[SourceSpec]:
    if not values:
        return list(SOURCES.values())
    unknown = sorted(set(values) - set(SOURCES))
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(unknown)}")
    return [SOURCES[name] for name in values]


def empty_source_stats() -> dict:
    return {
        "examined": {"documents": 0, "tokens": 0, "characters": 0},
        "steps": {
            name: {"documents": 0, "tokens": 0, "characters": 0}
            for name in FILTER_NAMES
        },
        "rejected": {"documents": 0, "tokens": 0},
        "rejected_reasons": Counter(),
        "accepted": {"documents": 0, "tokens": 0, "characters": 0},
    }


def add_step(stats: dict, name: str, tokens: int, characters: int) -> None:
    stats["steps"][name]["documents"] += 1
    stats["steps"][name]["tokens"] += tokens
    stats["steps"][name]["characters"] += characters


def reject(stats: dict, reason: str, tokens: int) -> None:
    stats["rejected"]["documents"] += 1
    stats["rejected"]["tokens"] += tokens
    stats["rejected_reasons"][reason] += 1


def round_robin(
    paths: dict[str, Path],
) -> Iterator[tuple[SourceSpec, dict]]:
    handles = {
        name: read_jsonl(path)
        for name, path in paths.items()
        if path.is_file()
    }
    missing = sorted(set(paths) - set(handles))
    if missing:
        print(f"WARNING: raw files missing for {missing}", file=sys.stderr)
    while handles:
        for name in list(handles):
            try:
                record = next(handles[name])
            except StopIteration:
                handles.pop(name)
                continue
            yield SOURCES[name], record


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Stage B raw corpus.")
    parser.add_argument("--input-dir", default="data/v8/pretrain/raw")
    parser.add_argument("--output-dir", default="data/v8/pretrain/filtered")
    parser.add_argument("--tokenizer", default="data/v8/tokenizer.json")
    parser.add_argument(
        "--leak-index",
        default="data/v8/pretrain/leak/leak_index.jsonl",
    )
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--min-lang-score-fr", type=float, default=0.04)
    parser.add_argument("--min-lang-score-en", type=float, default=0.06)
    parser.add_argument("--minhash-hashes", type=int, default=12)
    parser.add_argument("--minhash-bands", type=int, default=4)
    parser.add_argument("--minhash-threshold", type=float, default=0.80)
    parser.add_argument("--minhash-max-words", type=int, default=20_000)
    parser.add_argument("--skip-minhash", action="store_true")
    parser.add_argument("--skip-token-counts", action="store_true")
    parser.add_argument("--allow-missing-leak", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_docs is not None and args.max_docs < 1:
        parser.error("--max-docs must be positive.")
    if not 1 <= args.minhash_bands <= args.minhash_hashes:
        parser.error("--minhash-bands must be in [1, --minhash-hashes].")
    if args.minhash_hashes % args.minhash_bands != 0:
        parser.error("--minhash-hashes must be divisible by --minhash-bands.")

    specs = parse_sources(args.sources)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = TokenizerWrapper(tokenizer_path)

    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists. Choose a new output directory; "
            "this script never overwrites a corpus."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)

    leak_path = Path(args.leak_index)
    if leak_path.is_file():
        leak_records = list(read_jsonl(leak_path))
        leak_index = LeakIndex(
            leak_records,
            num_hashes=args.minhash_hashes,
            bands=args.minhash_bands,
            seed=args.seed,
        )
        leak_stats = {
            "path": str(leak_path),
            "records": leak_index.records,
            "texts": leak_index.texts,
            "exact_hashes": len(leak_index.exact),
            "containment_shingles": len(leak_index.containment),
            "lsh_bands": len(leak_index.band_index),
        }
    elif args.allow_missing_leak:
        leak_index = LeakIndex(
            [],
            num_hashes=args.minhash_hashes,
            bands=args.minhash_bands,
            seed=args.seed,
        )
        leak_stats = {"path": str(leak_path), "missing": True, "records": 0}
        print(
            "WARNING: leak index is missing; anti-leak checks are disabled.",
            file=sys.stderr,
        )
    else:
        raise SystemExit(
            f"Leak index not found: {leak_path}. Run "
            "scripts/build_leak_index_v8.py first, or pass "
            "--allow-missing-leak for a smoke test."
        )

    params = make_minhash_params(args.minhash_hashes, args.seed)
    rows = args.minhash_hashes // args.minhash_bands
    signatures: list[np.ndarray] = []
    bands: dict[tuple[int, ...], list[int]] = {}
    seen: set[bytes] = set()
    source_stats = {spec.name: empty_source_stats() for spec in specs}
    handles = {}
    examined_docs = 0
    examined_tokens = 0
    started = time.time()
    chars_per_token = {spec.name: None for spec in specs}

    paths = {spec.name: input_dir / f"{spec.name}.jsonl" for spec in specs}

    try:
        for spec, record in round_robin(paths):
            if args.max_docs is not None and examined_docs >= args.max_docs:
                break
            stats = source_stats[spec.name]
            raw_text = record.get("text")
            if not isinstance(raw_text, str):
                continue

            text = normalize_text(raw_text, spec.kind)
            characters = len(text)
            if args.skip_token_counts:
                tokens = 0
            else:
                ids = tokenizer.encode(text)
                if 0 in ids:
                    examined_docs += 1
                    stats["examined"]["documents"] += 1
                    stats["examined"]["characters"] += characters
                    reject(stats, "eos_id_in_encoded_text", 0)
                    continue
                tokens = len(ids)

            examined_docs += 1
            examined_tokens += tokens
            stats["examined"]["documents"] += 1
            stats["examined"]["tokens"] += tokens
            stats["examined"]["characters"] += characters

            if characters < args.min_chars or characters > args.max_chars:
                reject(
                    stats,
                    "too_short" if characters < args.min_chars else "too_long",
                    tokens,
                )
                continue
            add_step(stats, "length", tokens, characters)

            if spec.lang is not None:
                minimum = (
                    args.min_lang_score_fr
                    if spec.lang == "fr"
                    else args.min_lang_score_en
                )
                ok, _score = language_ok(text, spec.lang, minimum)
                if not ok:
                    reject(stats, "language", tokens)
                    continue
            add_step(stats, "language", tokens, characters)

            if alpha_ratio(text) < spec.min_alpha_ratio:
                reject(stats, "alphabet_ratio", tokens)
                continue
            add_step(stats, "alphabet", tokens, characters)

            words = sample_words(word_tokens(text), args.minhash_max_words)
            shingles5 = shingle_hashes_from_words(words, 5)
            if len(shingles5):
                repetition = 1.0 - len(np.unique(shingles5)) / len(shingles5)
            else:
                repetition = 0.0
            if repetition > spec.max_repetition:
                reject(stats, "repetition", tokens)
                continue
            add_step(stats, "repetition", tokens, characters)

            digest = sha1_digest(text)
            signature = minhash_signature(shingles5, params)

            # Exact dedup
            if digest in seen:
                reject(stats, "duplicate_exact", tokens)
                continue
            add_step(stats, "exact_duplicate", tokens, characters)

            duplicate = False
            if not args.skip_minhash:
                for key in band_keys(signature, rows):
                    for other_index in bands.get(key, ()):
                        if (
                            jaccard_estimate(signature, signatures[other_index])
                            >= args.minhash_threshold
                        ):
                            duplicate = True
                            break
                    if duplicate:
                        break
            if duplicate:
                reject(stats, "near_duplicate", tokens)
                continue
            add_step(stats, "near_duplicate", tokens, characters)

            leak_reason = leak_index.check(digest, words, signature)
            if leak_reason is not None:
                reject(stats, leak_reason, tokens)
                continue
            add_step(stats, "leak", tokens, characters)

            if digest in seen:
                reject(stats, "duplicate_exact", tokens)
                continue

            if not args.skip_minhash:
                index = len(signatures)
                signatures.append(signature)
                for key in band_keys(signature, rows):
                    bands.setdefault(key, []).append(index)

            seen.add(digest)
            split = split_for_text(text, VAL_FRAC, TEST_FRAC)

            if spec.name not in handles:
                handles[spec.name] = (staging / f"{spec.name}.jsonl").open(
                    "w", encoding="utf-8"
                )
            handle = handles[spec.name]
            handle.write(
                json.dumps(
                    {
                        "id": digest.hex(),
                        "source": spec.name,
                        "lang": spec.lang,
                        "kind": spec.kind,
                        "text": text,
                        "characters": characters,
                        "tokens": tokens,
                        "split": split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["accepted"]["documents"] += 1
            stats["accepted"]["tokens"] += tokens
            stats["accepted"]["characters"] += characters
            if tokens:
                chars_per_token[spec.name] = (
                    stats["accepted"]["characters"] / stats["accepted"]["tokens"]
                )

            if examined_docs % 1000 == 0:
                elapsed = max(1e-9, time.time() - started)
                print(
                    f"[filter] {examined_docs:,} docs examined / "
                    f"{examined_tokens:,} tokens / "
                    f"{examined_docs / elapsed:.1f} docs/s",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    for stats in source_stats.values():
        stats["rejected_reasons"] = counter_dict(stats["rejected_reasons"])
        stats["accepted"]["chars_per_token"] = (
            round(stats["accepted"]["characters"] / stats["accepted"]["tokens"], 3)
            if stats["accepted"]["tokens"]
            else None
        )

    total_accepted_docs = sum(s["accepted"]["documents"] for s in source_stats.values())
    total_accepted_tokens = sum(s["accepted"]["tokens"] for s in source_stats.values())
    summary = {
        "format_version": 1,
        "stage": "filter",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "thresholds": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_lang_score_fr": args.min_lang_score_fr,
            "min_lang_score_en": args.min_lang_score_en,
            "minhash_hashes": args.minhash_hashes,
            "minhash_bands": args.minhash_bands,
            "minhash_threshold": args.minhash_threshold,
            "minhash_max_words": args.minhash_max_words,
            "skip_minhash": args.skip_minhash,
            "skip_token_counts": args.skip_token_counts,
        },
        "leak_index": leak_stats,
        "sources": source_stats,
        "totals": {
            "examined_documents": examined_docs,
            "examined_tokens": examined_tokens,
            "accepted_documents": total_accepted_docs,
            "accepted_tokens": total_accepted_tokens,
            "accepted_characters": sum(
                s["accepted"]["characters"] for s in source_stats.values()
            ),
            "rejected_documents": examined_docs - total_accepted_docs,
            "rejected_tokens": examined_tokens - total_accepted_tokens,
        },
        "elapsed_s": round(time.time() - started, 3),
    }
    json_dump(staging / "filter_stats.json", summary)
    staging.replace(output_dir)
    print(json.dumps(summary["totals"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
