#!/usr/bin/env python3
"""Validate the v8 tokenizer against the acceptance criteria.

Criteria checked:

* round-trip is lossless after NFC normalisation;
* the four special tokens exist and keep their roles;
* ``é``, ``è``, ``à`` and ``ç`` are each encoded as exactly one token;
* no standalone ``Ġ`` token is emitted before a word;
* no learned token merges two whitespace-separated words;
* French held-out text uses at most 1.5 tokens per whitespace word.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from tinyllm.tokenizer import (
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    TokenizerWrapper,
)

ACCENTS = ("é", "è", "à", "ç")

ROUNDTRIP_TEXTS = [
    "Bonjour, comment vas-tu aujourd'hui ?",
    "La France est un pays d'Europe. Sa capitale est Paris.",
    "é è à ç ù â ê î ô û ë ï ü",
    "def hello():\n    return 42\n",
    "The capital of France is Paris.",
]


def read_lines(path: Path | None, limit: int = 200) -> list[str]:
    if path is None or not path.is_file():
        return []
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(line)
            if len(documents) >= limit:
                break
    return documents


def tokens_per_word(tokenizer: TokenizerWrapper, documents: list[str]) -> float:
    tokens = sum(len(tokenizer.encode(document)) for document in documents)
    words = sum(len(document.split()) for document in documents)
    if words == 0:
        raise ValueError("The evaluation sample contains no words.")
    return tokens / words


def check_roundtrip(tokenizer: TokenizerWrapper) -> list[str]:
    failures: list[str] = []
    for text in ROUNDTRIP_TEXTS:
        expected = unicodedata.normalize("NFC", text)
        decoded = tokenizer.decode(tokenizer.encode(text))
        if decoded != expected:
            failures.append(f"round-trip mismatch: {text!r} -> {decoded!r}")
    return failures


def check_accents(tokenizer: TokenizerWrapper) -> list[str]:
    failures: list[str] = []
    for char in ACCENTS:
        ids = tokenizer.encode(char)
        if len(ids) != 1:
            tokens = [tokenizer._tok.id_to_token(token_id) for token_id in ids]
            failures.append(f"{char!r} is {len(ids)} tokens, not 1: {tokens}")
    return failures


def check_special_tokens(tokenizer: TokenizerWrapper) -> list[str]:
    failures: list[str] = []
    expected = {
        "eos": (tokenizer.eos_id, EOS_TOKEN),
        "pad": (tokenizer.pad_id, PAD_TOKEN),
        "user": (tokenizer.user_id, USER_TOKEN),
        "assistant": (tokenizer.assistant_id, ASSISTANT_TOKEN),
    }
    for name, (token_id, token) in expected.items():
        if tokenizer._tok.token_to_id(token) != token_id:
            failures.append(f"{name} token id mismatch")
        if tokenizer._tok.id_to_token(token_id) != token:
            failures.append(f"{name} id {token_id} does not decode to {token!r}")
    if tokenizer.eos_id != 0:
        failures.append(f"EOS id is {tokenizer.eos_id}, expected 0 for v8")
    return failures


def check_no_isolated_space(
    tokenizer: TokenizerWrapper,
    documents: list[str],
) -> tuple[list[str], dict]:
    failures: list[str] = []
    pieces = 0
    isolated = 0
    for document in documents:
        pre_tokenized = tokenizer._tok.pre_tokenizer.pre_tokenize_str(document)
        pieces += len(pre_tokenized)
        for piece, _ in pre_tokenized:
            if piece == "Ġ":
                isolated += 1
                if len(failures) < 5:
                    failures.append(
                        f"standalone Ġ piece in: {document[:80]!r}"
                    )

    encoded_isolated = 0
    for document in documents:
        for token_id in tokenizer.encode(document):
            if tokenizer._tok.id_to_token(token_id) == "Ġ":
                encoded_isolated += 1
    if encoded_isolated:
        failures.append(
            f"{encoded_isolated} standalone Ġ tokens in encoded held-out text"
        )

    return failures, {
        "pre_tokenized_pieces": pieces,
        "standalone_space_pieces": isolated,
        "standalone_space_tokens": encoded_isolated,
    }


def check_no_cross_word_merges(tokenizer: TokenizerWrapper) -> list[str]:
    failures: list[str] = []
    vocab = tokenizer._tok.get_vocab()
    for token, _ in vocab.items():
        # A leading run of space/newline/tab markers is indentation.  A space
        # marker after a non-whitespace character would be a cross-word merge.
        body = token.lstrip("ĠĊĉ")
        if "Ġ" in body:
            if len(failures) < 10:
                failures.append(f"token with internal space marker: {token!r}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the v8 tokenizer.")
    parser.add_argument("--tokenizer", default="data/v8/tokenizer.json")
    parser.add_argument(
        "--fr-eval",
        default="data/v8/tokenizer_corpus/heldout_fr.txt",
    )
    parser.add_argument(
        "--en-eval",
        default="data/v8/tokenizer_corpus/heldout_en.txt",
    )
    parser.add_argument("--max-fr-tokens-per-word", type=float, default=1.5)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Optional exact vocabulary size check.",
    )
    args = parser.parse_args()

    tokenizer = TokenizerWrapper(args.tokenizer)
    fr_documents = read_lines(Path(args.fr_eval))
    en_documents = read_lines(Path(args.en_eval))

    failures: list[str] = []
    failures.extend(check_roundtrip(tokenizer))
    failures.extend(check_accents(tokenizer))
    failures.extend(check_special_tokens(tokenizer))
    failures.extend(check_no_cross_word_merges(tokenizer))

    space_failures, space_stats = check_no_isolated_space(
        tokenizer,
        fr_documents or [ROUNDTRIP_TEXTS[0]],
    )
    failures.extend(space_failures)

    if args.vocab_size is not None and tokenizer.vocab_size != args.vocab_size:
        failures.append(
            f"vocab_size is {tokenizer.vocab_size}, expected {args.vocab_size}"
        )

    summary = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "fr_eval_documents": len(fr_documents),
        "en_eval_documents": len(en_documents),
        "fr_tokens_per_word": None,
        "en_tokens_per_word": None,
        **space_stats,
        "failures": failures,
        "ok": not failures,
    }

    if fr_documents:
        fr_ratio = tokens_per_word(tokenizer, fr_documents)
        summary["fr_tokens_per_word"] = round(fr_ratio, 3)
        if fr_ratio > args.max_fr_tokens_per_word:
            failures.append(
                f"French tokens/word is {fr_ratio:.3f} > "
                f"{args.max_fr_tokens_per_word}"
            )
    else:
        failures.append(f"French evaluation sample is empty: {args.fr_eval}")

    if en_documents:
        summary["en_tokens_per_word"] = round(
            tokens_per_word(tokenizer, en_documents), 3
        )

    summary["ok"] = not failures
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
