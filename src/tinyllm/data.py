import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def dedupe(docs: Iterable[dict], seen: set[str] | None = None) -> Iterator[dict]:
    seen = set() if seen is None else seen
    for doc in docs:
        digest = text_hash(doc["text"])
        if digest in seen:
            continue
        seen.add(digest)
        yield doc


def split_documents(
    docs: Iterable[dict], val_frac: float = 0.01, test_frac: float = 0.01
) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for doc in docs:
        position = int(text_hash(doc["text"])[:8], 16) / 0xFFFFFFFF
        if position < test_frac:
            splits["test"].append(doc)
        elif position < test_frac + val_frac:
            splits["val"].append(doc)
        else:
            splits["train"].append(doc)
    return splits


def split_for(text: str, val_frac: float, test_frac: float) -> str:
    position = int(text_hash(text)[:8], 16) / 0xFFFFFFFF
    if position < test_frac:
        return "test"
    if position < test_frac + val_frac:
        return "val"
    return "train"


def pack_documents(
    documents: Iterable[list[int]], context_len: int, eos_id: int
) -> Iterator[list[int]]:
    buffer: list[int] = []
    for document in documents:
        buffer.extend(document)
        buffer.append(eos_id)
        while len(buffer) >= context_len:
            yield buffer[:context_len]
            buffer = buffer[context_len:]


def write_bin(chunks: Iterable[list[int]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("wb") as handle:
        for chunk in chunks:
            np.asarray(chunk, dtype=np.uint16).tofile(handle)
            written += len(chunk)
    return written


def read_bin(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size == 0:
        return np.empty(0, dtype=np.uint16)
    return np.memmap(path, dtype=np.uint16, mode="r")


class PackedWriter:
    def __init__(self, path: str | Path, context_len: int, eos_id: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context_len = context_len
        self.eos_id = eos_id
        self._buffer: list[int] = []
        self._handle = self.path.open("wb")
        self.tokens_written = 0

    def add_document(self, token_ids: list[int]) -> None:
        self._buffer.extend(token_ids)
        self._buffer.append(self.eos_id)
        self._flush()

    def _flush(self) -> None:
        while len(self._buffer) >= self.context_len:
            chunk = self._buffer[: self.context_len]
            np.asarray(chunk, dtype=np.uint16).tofile(self._handle)
            self.tokens_written += self.context_len
            self._buffer = self._buffer[self.context_len :]

    def close(self) -> int:
        self._handle.close()
        return self.tokens_written
