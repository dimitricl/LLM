import hashlib
import re
from collections.abc import Iterable, Iterator

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
