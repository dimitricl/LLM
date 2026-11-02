import argparse
import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from tinyllm.config import RunConfig, load_config
from tinyllm.data import PackedWriter, clean_text, split_for, text_hash
from tinyllm.tokenizer import TokenizerWrapper

SOURCES = {
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "fineweb_edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "wiki_en": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "wiki_fr": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.fr",
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "python": {
        "name": "bigcode/the-stack-smol",
        "config": None,
        "split": "train",
        "text_field": "content",
        "data_dir": "data/python",
    },
}


def iter_hf_documents(source_name: str) -> Iterator[dict]:
    from datasets import load_dataset

    spec = SOURCES[source_name]
    dataset = load_dataset(
        spec["name"],
        spec["config"],
        split=spec["split"],
        streaming=True,
        data_dir=spec["data_dir"],
    )
    for row in dataset:
        text = row[spec["text_field"]]
        if text:
            yield {"text": text}


def iter_mixed_documents(
    mix: dict[str, float], seed: int, max_docs: int | None = None
) -> Iterator[tuple[str, dict]]:
    import random

    rng = random.Random(seed)
    names = list(mix)
    weights = [mix[name] for name in names]
    streams = {name: iter_hf_documents(name) for name in names}
    produced = 0
    while names and (max_docs is None or produced < max_docs):
        name = rng.choices(names, weights=weights, k=1)[0]
        try:
            document = next(streams[name])
        except StopIteration:
            index = names.index(name)
            names.pop(index)
            weights.pop(index)
            streams.pop(name)
            continue
        produced += 1
        yield name, document


def prepare_data(
    cfg: RunConfig,
    document_stream: Callable[[], Iterator[tuple[str, dict]]] | None = None,
    max_docs: int | None = None,
    force: bool = False,
) -> dict:
    train_path = Path(cfg.data.train_bin)
    meta_path = train_path.with_name("meta.json")
    if not force and train_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    if document_stream is None:
        document_stream = lambda: iter_mixed_documents(cfg.data.sources, cfg.train.seed)

    tokenizer = TokenizerWrapper(cfg.data.tokenizer_path)
    writers = {
        "train": PackedWriter(train_path, cfg.model.context_len, tokenizer.eos_id),
        "val": PackedWriter(cfg.data.val_bin, cfg.model.context_len, tokenizer.eos_id),
        "test": PackedWriter(cfg.data.test_bin, cfg.model.context_len, tokenizer.eos_id),
    }
    seen: set[str] = set()
    docs = 0
    tokens_by_source: dict[str, int] = {}
    for source_name, document in document_stream():
        text = clean_text(document["text"])
        if not (cfg.data.min_chars <= len(text) <= cfg.data.max_chars):
            continue
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        split = split_for(text, cfg.data.val_frac, cfg.data.test_frac)
        token_ids = tokenizer.encode(text)
        if not token_ids:
            continue
        writers[split].add_document(token_ids)
        tokens_by_source[source_name] = (
            tokens_by_source.get(source_name, 0) + len(token_ids) + 1
        )
        docs += 1
        if max_docs is not None and docs >= max_docs:
            break

    tokens = {name: writer.close() for name, writer in writers.items()}
    meta = {
        "sources": cfg.data.sources,
        "tokens": tokens,
        "tokens_by_source": tokens_by_source,
        "docs": docs,
        "context_len": cfg.model.context_len,
        "tokenizer_sha1": hashlib.sha1(
            Path(cfg.data.tokenizer_path).read_bytes()
        ).hexdigest(),
        "seed": cfg.train.seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    meta = prepare_data(cfg, max_docs=args.max_docs, force=args.force)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
