#!/usr/bin/env python3
"""Build a balanced tokenizer-training corpus from streaming raw sources.

The v8 tokenizer must be trained on *raw text*, not on chat conversations.
This script streams the same families of sources that will be used for
pretraining, applies light quality filters, and writes an interleaved JSONL
corpus plus two held-out samples used by ``scripts/test_tokenizer_v8.py``.

Mix (fractions of the character budget):

* French 50 %: Wikipedia FR 30 %, FineWeb-2 ``fra_Latn`` 20 %
  (FineWeb-2 is therefore capped at 40 % of the French budget);
* English 45 %: FineWeb-Edu 37.8 %, Wikipedia EN 5.85 %, TinyStories 1.35 %
  (TinyStories is capped at 3 % of the English budget);
* Code 5 %: CodeParrot Clean (raw Python files).

No file is downloaded in full: Hugging Face datasets are used in streaming
mode.  The output directory is created atomically and never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from tinyllm.data import clean_text

FR_STOPWORDS = set(
    "le la les de des du un une et est en que qui pour dans pas au aux ce "
    "cette il elle nous vous ils elles avec sur ne se son sa ses mais ou "
    "où donc car comme plus moins très tout tous toute toutes être avoir "
    "fait faire dit aussi bien deux autre même".split()
)
EN_STOPWORDS = set(
    "the a an and is are was were to of in for on with that this it as at "
    "by from be or not have has had do does did but if then than so we you "
    "they he she i".split()
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_ACCENTS = set("àâçéèêëîïôûùüÿœæ")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    field: str
    kind: str  # "text" or "code"
    lang: str | None  # "fr", "en" or None
    fraction: float


SOURCES = (
    SourceSpec(
        "fr_wikipedia",
        "wikimedia/wikipedia",
        "20231101.fr",
        "train",
        "text",
        "text",
        "fr",
        0.30,
    ),
    SourceSpec(
        "fr_fineweb2",
        "HuggingFaceFW/fineweb-2",
        "fra_Latn",
        "train",
        "text",
        "text",
        "fr",
        0.20,
    ),
    SourceSpec(
        "en_fineweb_edu",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
        "text",
        "text",
        "en",
        0.378,
    ),
    SourceSpec(
        "en_wikipedia",
        "wikimedia/wikipedia",
        "20231101.en",
        "train",
        "text",
        "text",
        "en",
        0.0585,
    ),
    SourceSpec(
        "en_tinystories",
        "roneneldan/TinyStories",
        None,
        "train",
        "text",
        "text",
        "en",
        0.0135,
    ),
    SourceSpec(
        "code_python",
        "codeparrot/codeparrot-clean",
        None,
        "train",
        "content",
        "code",
        None,
        0.05,
    ),
)


def iter_source(spec: SourceSpec, seed: int) -> Iterator[str]:
    """Stream one Hugging Face source, shuffled with a bounded buffer."""
    kwargs = {"split": spec.split, "streaming": True}
    if spec.config is not None:
        kwargs["name"] = spec.config
    dataset = load_dataset(spec.dataset, **kwargs)
    # A large shuffle buffer makes the first document wait for thousands of
    # upstream rows; 1 000 is enough to mix shards without a long warm-up.
    dataset = dataset.shuffle(seed=seed, buffer_size=1_000)
    for row in dataset:
        value = row.get(spec.field)
        if isinstance(value, str) and value.strip():
            yield value


def clean_source_text(text: str, kind: str) -> str:
    if kind == "code":
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()
    return clean_text(text)


def alpha_ratio(text: str) -> float:
    non_space = sum(1 for char in text if not char.isspace())
    if non_space == 0:
        return 0.0
    return sum(1 for char in text if char.isalpha()) / non_space


def language_ok(text: str, lang: str) -> bool:
    words = [word.lower() for word in _WORD.findall(text)]
    if len(words) < 20:
        return False
    if lang == "fr":
        stop = sum(word in FR_STOPWORDS for word in words) / len(words)
        accents = sum(char in _ACCENTS for char in text) / max(1, len(text))
        return stop >= 0.035 or (stop >= 0.02 and accents >= 0.004)
    if lang == "en":
        stop = sum(word in EN_STOPWORDS for word in words) / len(words)
        return stop >= 0.035
    return True


def is_repetitive(text: str, kind: str) -> bool:
    if kind == "code":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 20:
            return False
        counts = Counter(lines)
        return counts.most_common(1)[0][1] / len(lines) > 0.5

    words = [word.lower() for word in _WORD.findall(text)]
    if len(words) < 60:
        return False
    if len(set(words)) / len(words) < 0.25:
        return True
    grams = Counter(tuple(words[index : index + 5]) for index in range(len(words) - 4))
    return max(grams.values()) / max(1, sum(grams.values())) > 0.20


def is_usable(
    text: str,
    spec: SourceSpec,
    min_chars: int,
    max_chars: int,
    min_alpha: float,
) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    if spec.kind == "code":
        if alpha_ratio(text) < 0.15:
            return False
    else:
        if alpha_ratio(text) < min_alpha:
            return False
        if spec.lang is not None and not language_ok(text, spec.lang):
            return False
    return not is_repetitive(text, spec.kind)


def collect_heldout(
    iterator: Iterator[str],
    spec: SourceSpec,
    count: int,
    min_chars: int,
    max_chars: int,
    min_alpha: float,
) -> list[str]:
    documents: list[str] = []
    for raw in iterator:
        text = clean_source_text(raw, spec.kind)[:max_chars]
        if not is_usable(text, spec, min_chars, max_chars, min_alpha):
            continue
        documents.append(text)
        if len(documents) >= count:
            break
    return documents


def build_corpus(
    output_dir: Path,
    max_mb: int,
    heldout_docs: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    min_alpha: float,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Choose a new output directory; nothing was overwritten."
        )

    total_chars = int(max_mb * 1024 * 1024)
    targets = {
        spec.name: int(total_chars * spec.fraction) for spec in SOURCES
    }
    iterators = {spec.name: iter_source(spec, seed) for spec in SOURCES}
    chars: Counter = Counter()
    docs: Counter = Counter()
    discarded: Counter = Counter()
    started = time.perf_counter()
    last_report = 0
    print(
        f"streaming {len(SOURCES)} sources for {total_chars / 1e6:.0f} M chars",
        flush=True,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    corpus_path = staging / "corpus.jsonl"

    try:
        with corpus_path.open("w", encoding="utf-8") as handle:
            active = {spec.name for spec in SOURCES}
            specs = {spec.name: spec for spec in SOURCES}

            while active:
                for name in list(active):
                    spec = specs[name]
                    if chars[name] >= targets[name]:
                        active.discard(name)
                        continue

                    while True:
                        try:
                            raw = next(iterators[name])
                        except StopIteration:
                            active.discard(name)
                            break

                        text = clean_source_text(raw, spec.kind)[:max_chars]
                        if not is_usable(
                            text,
                            spec,
                            min_chars,
                            max_chars,
                            min_alpha,
                        ):
                            discarded[name] += 1
                            continue

                        remaining = targets[name] - chars[name]
                        if len(text) > remaining:
                            text = text[:remaining].rstrip()
                        # When fewer than min_chars remain, keep the small tail
                        # instead of repeatedly pulling documents forever.
                        if len(text) < min_chars and remaining >= min_chars:
                            discarded[name] += 1
                            continue

                        handle.write(
                            json.dumps(
                                {"source": name, "text": text},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        chars[name] += len(text)
                        docs[name] += 1
                        written = sum(docs.values())
                        if written >= last_report + 1_000:
                            last_report = written
                            print(
                                f"  {written:,} docs / "
                                f"{sum(chars.values()) / 1e6:.1f} M chars",
                                flush=True,
                            )
                        break

        # Held-out documents are pulled *after* the training budget: they are
        # never written to corpus.jsonl and can be used for tokenizer tests.
        heldout_sources = {
            "fr": ("fr_wikipedia", "fr_fineweb2"),
            "en": ("en_fineweb_edu", "en_wikipedia"),
        }
        heldout: dict[str, list[str]] = {}
        for lang, names in heldout_sources.items():
            documents: list[str] = []
            per_source = max(1, heldout_docs // len(names))
            for name in names:
                documents.extend(
                    collect_heldout(
                        iterators[name],
                        specs[name],
                        per_source,
                        min_chars,
                        max_chars,
                        min_alpha,
                    )
                )
            heldout[lang] = documents
            heldout_path = staging / f"heldout_{lang}.txt"
            with heldout_path.open("w", encoding="utf-8") as handle:
                for document in documents:
                    handle.write(" ".join(document.split()) + "\n")

        elapsed = time.perf_counter() - started
        manifest = {
            "format_version": 1,
            "kind": "tokenizer_training_corpus",
            "seed": seed,
            "max_mb": max_mb,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_alpha_ratio": min_alpha,
            "sources": {
                spec.name: {
                    "dataset": spec.dataset,
                    "config": spec.config,
                    "split": spec.split,
                    "field": spec.field,
                    "kind": spec.kind,
                    "language": spec.lang,
                    "fraction": spec.fraction,
                    "target_chars": targets[spec.name],
                    "written_chars": chars[spec.name],
                    "documents": docs[spec.name],
                    "discarded_documents": discarded[spec.name],
                }
                for spec in SOURCES
            },
            "totals": {
                "written_chars": sum(chars.values()),
                "documents": sum(docs.values()),
                "heldout_documents": {
                    lang: len(documents) for lang, documents in heldout.items()
                },
            },
            "elapsed_s": elapsed,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced raw-text tokenizer corpus."
    )
    parser.add_argument(
        "--output-dir",
        default="data/v8/tokenizer_corpus",
    )
    parser.add_argument("--max-mb", type=int, default=300)
    parser.add_argument("--heldout-docs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.55)
    args = parser.parse_args()

    if args.max_mb < 1 or args.heldout_docs < 1:
        parser.error("--max-mb and --heldout-docs must be positive.")
    if not 0.0 <= args.min_alpha_ratio <= 1.0:
        parser.error("--min-alpha-ratio must be in [0, 1].")

    manifest = build_corpus(
        output_dir=Path(args.output_dir),
        max_mb=args.max_mb,
        heldout_docs=args.heldout_docs,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_alpha=args.min_alpha_ratio,
    )
    print(json.dumps(manifest["totals"], indent=2))
    print(f"corpus written to {args.output_dir}")


if __name__ == "__main__":
    main()
