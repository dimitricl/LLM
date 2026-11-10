#!/usr/bin/env python3
"""Train the v8 byte-level BPE tokenizer on the balanced raw corpus."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tinyllm.tokenizer import V8_VOCAB_SIZE, train_tokenizer_v8


def iter_texts(path: Path, max_chars: int | None) -> list[str]:
    texts: list[str] = []
    consumed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str) or not text:
                continue
            if max_chars is not None:
                remaining = max_chars - consumed
                if remaining <= 0:
                    break
                text = text[:remaining]
            texts.append(text)
            consumed += len(text)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TinyLLM v8 tokenizer.")
    parser.add_argument(
        "--corpus",
        default="data/v8/tokenizer_corpus/corpus.jsonl",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=V8_VOCAB_SIZE,
    )
    parser.add_argument("--out", default="data/v8/tokenizer.json")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional cap on training characters.",
    )
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")

    started = time.perf_counter()
    texts = iter_texts(corpus, args.max_chars)
    total_chars = sum(len(text) for text in texts)
    print(
        f"training on {len(texts):,} documents / {total_chars / 1e6:.1f} M chars",
        flush=True,
    )

    tokenizer = train_tokenizer_v8(
        texts,
        vocab_size=args.vocab_size,
        out_path=args.out,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "documents": len(texts),
                "chars": total_chars,
                "elapsed_s": round(elapsed, 2),
                "out": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
