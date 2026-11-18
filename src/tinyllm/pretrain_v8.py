"""Shared building blocks for the v8 pre-training corpus.

The module is deliberately dependency-light: it only needs numpy, xxhash and
the project tokenizer.  It implements the filters, exact dedup, MinHash
near-duplicate detection, leak detection, document splitting and the flat
uint16 binary writer used by the Stage B scripts.

Design rules enforced here
--------------------------
* EOS is the special token ``""`` with id ``0``.  It is appended by the
  tokenizer script, never obtained from ``TokenizerWrapper.encode``.
* A document is never truncated.  Length filters reject the whole document.
* A document is assigned to exactly one split (train/val/test) from a hash of
  its normalised text.
* The flat ``.bin`` stream contains every token of every accepted document,
  each document followed by exactly one EOS id.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import xxhash

EOS_ID = 0
PAD_ID = 1
CONTEXT_LEN = 512
TOKEN_DTYPE = np.uint16

#: 300 M target + 3.3 % evaluation reserve.  Train gets ~300.7 M tokens,
#: val and test ~4.65 M each.
TARGET_TOTAL_TOKENS = 310_000_000
VAL_FRAC = 0.015
TEST_FRAC = 0.015


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    field: str
    lang: str | None
    kind: str
    target_tokens: int
    char_budget: int
    min_alpha_ratio: float
    max_repetition: float

    @property
    def is_code(self) -> bool:
        return self.kind == "code"


SOURCES: dict[str, SourceSpec] = {
    "fr_wikipedia": SourceSpec(
        name="fr_wikipedia",
        dataset="wikimedia/wikipedia",
        config="20231101.fr",
        split="train",
        field="text",
        lang="fr",
        kind="text",
        target_tokens=93_000_000,
        char_budget=800_000_000,
        min_alpha_ratio=0.55,
        max_repetition=0.30,
    ),
    "fr_fineweb2": SourceSpec(
        name="fr_fineweb2",
        dataset="HuggingFaceFW/fineweb-2",
        config="fra_Latn",
        split="train",
        field="text",
        lang="fr",
        kind="text",
        target_tokens=62_000_000,
        char_budget=540_000_000,
        min_alpha_ratio=0.55,
        max_repetition=0.30,
    ),
    "en_fineweb_edu": SourceSpec(
        name="en_fineweb_edu",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        field="text",
        lang="en",
        kind="text",
        target_tokens=106_500_000,
        char_budget=990_000_000,
        min_alpha_ratio=0.55,
        max_repetition=0.30,
    ),
    "en_wikipedia": SourceSpec(
        name="en_wikipedia",
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        field="text",
        lang="en",
        kind="text",
        target_tokens=31_000_000,
        char_budget=290_000_000,
        min_alpha_ratio=0.55,
        max_repetition=0.30,
    ),
    "en_tinystories": SourceSpec(
        name="en_tinystories",
        dataset="roneneldan/TinyStories",
        config=None,
        split="train",
        field="text",
        lang="en",
        kind="text",
        target_tokens=2_000_000,
        char_budget=18_000_000,
        min_alpha_ratio=0.55,
        max_repetition=0.30,
    ),
    "code_python": SourceSpec(
        name="code_python",
        dataset="codeparrot/codeparrot-clean",
        config=None,
        split="train",
        field="content",
        lang=None,
        kind="code",
        target_tokens=15_500_000,
        char_budget=120_000_000,
        min_alpha_ratio=0.10,
        max_repetition=0.35,
    ),
}

LANG_TARGETS = {
    "fr": sum(s.target_tokens for s in SOURCES.values() if s.lang == "fr"),
    "en": sum(s.target_tokens for s in SOURCES.values() if s.lang == "en"),
    "code": sum(s.target_tokens for s in SOURCES.values() if s.lang is None),
}

# Sanity language check.  The HF configs already guarantee the language, so
# this only rejects grossly mislabelled or non-Latin documents.
FR_STOPWORDS = frozenset(
    """
    le la les de des du un une et est en que qui dans pour pas au aux ce cette
    ces il elle on nous vous ils elles avec sur par plus mais ou comme son sa
    ses leur leurs être avoir fait aussi très tout tous toute toutes sans sous
    entre chez lors donc car si ne y à d l qu n s c j m t o
    """.split()
)
EN_STOPWORDS = frozenset(
    """
    the and of to in is that for it with as was on are be this by from or an at
    not but they you we he she his her their have has had will would can could
    should there which what when where who how all more than about into over
    after before between because while if so no yes do does did been being its
    our your my me him them us i a
    """.split()
)
FR_ACCENTS = frozenset("àâäéèêëîïôöùûüÿçœæ")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_LINE_BREAK = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\f\v]+")
_BASE = np.uint64(0x9E3779B185EBCA87)
_WORD_HASH_CACHE: dict[str, int] = {}
_WORD_HASH_CACHE_LIMIT = 2_000_000


# ---------------------------------------------------------------------------
# Text normalisation and measurements
# ---------------------------------------------------------------------------


def normalize_text(text: str, kind: str = "text") -> str:
    """Normalise without ever cutting the document."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    if kind == "code":
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip("\n")
    text = unicodedata.normalize("NFC", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _LINE_BREAK.sub("\n\n", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def alpha_ratio(text: str) -> float:
    non_space = sum(1 for char in text if not char.isspace())
    if non_space == 0:
        return 0.0
    return sum(1 for char in text if char.isalpha()) / non_space


def language_score(text: str, lang: str) -> float:
    words = word_tokens(text)
    if not words:
        return 0.0
    if lang == "fr":
        hits = sum(word in FR_STOPWORDS for word in words)
        accents = sum(char in FR_ACCENTS for char in text.lower())
        return hits / len(words) + min(0.05, accents / len(words))
    if lang == "en":
        return sum(word in EN_STOPWORDS for word in words) / len(words)
    raise ValueError(f"Unsupported language: {lang}")


def language_ok(text: str, lang: str, min_score: float) -> tuple[bool, float]:
    score = language_score(text, lang)
    return score >= min_score, score


def _word_hash(word: str) -> int:
    value = _WORD_HASH_CACHE.get(word)
    if value is None:
        value = xxhash.xxh64(word.encode("utf-8")).intdigest()
        if len(_WORD_HASH_CACHE) < _WORD_HASH_CACHE_LIMIT:
            _WORD_HASH_CACHE[word] = value
    return value


def sample_words(words: list[str], max_words: int) -> list[str]:
    if max_words <= 0 or len(words) <= max_words:
        return words
    step = len(words) / max_words
    return [words[min(len(words) - 1, int(index * step))] for index in range(max_words)]


def shingle_hashes_from_words(words: list[str], n: int) -> np.ndarray:
    """Polynomial 64-bit hashes of every word n-gram."""
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    token_hashes = np.fromiter(
        (_word_hash(word) for word in words),
        dtype=np.uint64,
        count=len(words),
    )
    count = len(token_hashes) - n + 1
    hashes = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(n):
            hashes = hashes * _BASE + token_hashes[offset : offset + count]
    return hashes


def repetition_ratio(
    words: list[str],
    n: int = 5,
    max_words: int = 20_000,
) -> tuple[float, int]:
    """Fraction of duplicated word n-grams, on a bounded even sample."""
    sampled = sample_words(words, max_words)
    hashes = shingle_hashes_from_words(sampled, n)
    if len(hashes) == 0:
        return 0.0, len(words)
    unique = len(np.unique(hashes))
    return 1.0 - unique / len(hashes), len(words)


def sha1_digest(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()


def split_for_text(text: str, val_frac: float, test_frac: float) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_frac:
        return "test"
    if value < test_frac + val_frac:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------


def make_minhash_params(num_hashes: int = 12, seed: int = 42) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    params: list[tuple[int, int]] = []
    for _ in range(num_hashes):
        params.append((rng.getrandbits(64) | 1, rng.getrandbits(64)))
    return params


def minhash_signature(
    shingle_hashes: np.ndarray,
    params: list[tuple[int, int]],
) -> np.ndarray:
    if len(shingle_hashes) == 0:
        return np.zeros(len(params), dtype=np.uint64)
    signature = np.empty(len(params), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for index, (a, b) in enumerate(params):
            values = shingle_hashes * np.uint64(a) + np.uint64(b)
            signature[index] = values.min()
    return signature


def band_keys(signature: np.ndarray, rows: int) -> list[tuple[int, ...]]:
    if rows < 1 or len(signature) % rows != 0:
        raise ValueError("Signature length must be a multiple of rows.")
    return [
        tuple(int(value) for value in signature[start : start + rows])
        for start in range(0, len(signature), rows)
    ]


def jaccard_estimate(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) == 0:
        return 0.0
    return float(np.mean(left == right))


# ---------------------------------------------------------------------------
# Leak index
# ---------------------------------------------------------------------------


class LeakIndex:
    """Exact, near-duplicate and n-gram containment checks for eval leakage."""

    def __init__(
        self,
        records: Iterable[dict],
        *,
        num_hashes: int = 12,
        bands: int = 4,
        seed: int = 42,
        containment_n: int = 8,
        containment_samples: int = 256,
    ) -> None:
        self.params = make_minhash_params(num_hashes, seed)
        self.rows = max(1, num_hashes // bands)
        self.containment_n = containment_n
        self.containment_samples = containment_samples
        self.exact: set[bytes] = set()
        self.containment: set[int] = set()
        self.band_index: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.records = 0
        self.texts = 0

        for record in records:
            self.records += 1
            for text in leak_texts(record):
                normalized = normalize_text(text)
                if not normalized:
                    continue
                self.texts += 1
                self.exact.add(sha1_digest(normalized))
                words = sample_words(word_tokens(normalized), 20_000)
                shingles = shingle_hashes_from_words(words, containment_n)
                for value in shingles[:containment_samples]:
                    self.containment.add(int(value))
                signature = minhash_signature(
                    shingle_hashes_from_words(words, 5), self.params
                )
                for key in band_keys(signature, self.rows):
                    self.band_index.setdefault(key, []).append(signature)

    def check(
        self,
        digest: bytes,
        words: list[str],
        signature: np.ndarray,
    ) -> str | None:
        if digest in self.exact:
            return "leak_exact"
        for key in band_keys(signature, self.rows):
            for other in self.band_index.get(key, ()):
                if jaccard_estimate(signature, other) >= 0.80:
                    return "leak_near_duplicate"
        shingles = shingle_hashes_from_words(
            sample_words(words, 20_000), self.containment_n
        )
        if len(shingles):
            sample = shingles[: self.containment_samples]
            matched = sum(int(value) in self.containment for value in sample)
            if matched / len(sample) >= 0.50:
                return "leak_containment"
        return None


def leak_texts(record: dict) -> Iterator[str]:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                yield message["content"]
        return
    for key in ("text", "question", "answer", "content", "response"):
        value = record.get(key)
        if isinstance(value, str):
            yield value


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def append_stats(
    stats: dict,
    name: str,
    docs: int,
    tokens: int,
) -> None:
    stats["steps"].append(
        {"filter": name, "documents": docs, "tokens": tokens}
    )


# ---------------------------------------------------------------------------
# Flat binary writer
# ---------------------------------------------------------------------------


class FlatTokenWriter:
    """Write a flat uint16 stream; the final partial context is kept."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("wb")
        self.tokens = 0
        self.documents = 0
        self.zero_tokens = 0
        self.documents_longer_than_context = 0
        self.document_lengths: list[int] = []

    def add_document(self, ids: list[int], context_len: int = CONTEXT_LEN) -> None:
        if not ids or ids[-1] != EOS_ID:
            raise ValueError("A document must end with exactly one EOS id.")
        if EOS_ID in ids[:-1]:
            raise ValueError("EOS id found inside a document.")
        if len(ids) > 1:
            self.document_lengths.append(len(ids))
            if len(ids) > context_len:
                self.documents_longer_than_context += 1
        np.asarray(ids, dtype=TOKEN_DTYPE).tofile(self._handle)
        self.tokens += len(ids)
        self.documents += 1
        self.zero_tokens += 1

    def close(self) -> None:
        self._handle.close()

    def summary(self, context_len: int = CONTEXT_LEN) -> dict:
        windows = max(0, (self.tokens - 1) // context_len)
        covered = windows * context_len + 1
        return {
            "documents": self.documents,
            "tokens": self.tokens,
            "eos_tokens": self.zero_tokens,
            "mean_tokens_per_document": (
                round(self.tokens / self.documents, 3) if self.documents else 0.0
            ),
            "documents_longer_than_context": self.documents_longer_than_context,
            "documents_longer_than_context_pct": (
                round(100.0 * self.documents_longer_than_context / self.documents, 4)
                if self.documents
                else 0.0
            ),
            "documents_truncated": 0,
            "documents_truncated_pct": 0.0,
            "context_windows": windows,
            "tail_tokens_ignored_by_loader": max(0, self.tokens - covered),
        }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def count_tokens_in_bin(path: str | Path, value: int = EOS_ID) -> int:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    array = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
    total = 0
    chunk = 8_000_000
    for start in range(0, len(array), chunk):
        total += int((np.asarray(array[start : start + chunk]) == value).sum())
    return total


def bin_length(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size % np.dtype(TOKEN_DTYPE).itemsize != 0:
        raise ValueError(f"{path} is not uint16-aligned.")
    return size // np.dtype(TOKEN_DTYPE).itemsize


def percent(part: int, total: int) -> float:
    return round(100.0 * part / total, 3) if total else 0.0


def json_dump(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def counter_dict(counter: Counter) -> dict:
    return dict(counter.most_common())


def estimated_seconds(tokens: int, tokens_per_second: float) -> float:
    if tokens_per_second <= 0:
        return math.inf
    return tokens / tokens_per_second
