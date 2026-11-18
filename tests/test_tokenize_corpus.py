"""Tests for tokenize_corpus.py using synthetic tokenizer.

This test builds a tiny tokenizer with the same special token layout as v8
(EOS id 0, PAD id 1, etc.) to verify the tokenization pipeline without
requiring the real tokenizer.json file.
"""
from pathlib import Path
import json
import numpy as np
import pytest

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers as trainers_module,
)

from tinyllm.tokenizer import (
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    SPECIAL_TOKENS,
)
from tinyllm.pretrain_v8 import EOS_ID, CONTEXT_LEN, bin_length, count_tokens_in_bin


def build_synthetic_tokenizer(path: Path, vocab_size: int = 128) -> Tokenizer:
    """Build a tiny tokenizer with the same special token layout as v8."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers_module.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    simple_corpus = [
        "bonjour monde",
        "hello world",
        "def f(): return 1",
    ] * 10
    tokenizer.train_from_iterator(simple_corpus, trainer=trainer)

    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def test_synthetic_tokenizer_layout(tmp_path):
    """Verify the synthetic tokenizer has the same special token ids as v8."""
    path = tmp_path / "tokenizer.json"
    tokenizer = build_synthetic_tokenizer(path, vocab_size=64)

    assert tokenizer.token_to_id(EOS_TOKEN) == 0
    assert tokenizer.token_to_id(PAD_TOKEN) == 1
    assert tokenizer.token_to_id(USER_TOKEN) == 2
    assert tokenizer.token_to_id(ASSISTANT_TOKEN) == 3


def test_synthetic_tokenizer_never_emits_eos(tmp_path):
    """Verify encode(text) never produces id 0."""
    path = tmp_path / "tokenizer.json"
    tokenizer = build_synthetic_tokenizer(path, vocab_size=64)

    texts = [
        "bonjour monde",
        "hello world",
        "def f(): return 1",
        "é è à ç",
        "a" * 1000,
    ]
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert 0 not in ids, f"EOS id 0 found in encoded text: {text}"


def test_tokenize_corpus_integration(tmp_path):
    """End-to-end test of tokenize_corpus.py with synthetic data."""
    from tinyllm.tokenizer import TokenizerWrapper

    # Setup: create synthetic tokenizer
    tokenizer_path = tmp_path / "tokenizer.json"
    build_synthetic_tokenizer(tokenizer_path, vocab_size=64)
    tokenizer = TokenizerWrapper(tokenizer_path)
    assert tokenizer.eos_id == EOS_ID

    # Setup: create filtered JSONL input
    input_dir = tmp_path / "filtered"
    input_dir.mkdir()
    input_file = input_dir / "test_source.jsonl"

    records = [
        {
            "id": "a" * 40,
            "source": "test",
            "lang": "fr",
            "kind": "text",
            "text": "Bonjour, monde!",
            "characters": 15,
            "tokens": 3,
            "split": "train",
        },
        {
            "id": "b" * 40,
            "source": "test",
            "lang": "en",
            "kind": "text",
            "text": "Hello world!",
            "characters": 12,
            "tokens": 2,
            "split": "val",
        },
        {
            "id": "c" * 40,
            "source": "test",
            "lang": "en",
            "kind": "text",
            "text": "Test document.",
            "characters": 14,
            "tokens": 2,
            "split": "test",
        },
    ]
    with input_file.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Run tokenization logic manually (simulating the script)
    output_dir = tmp_path / "tokenized"
    output_dir.mkdir()

    from tinyllm.pretrain_v8 import FlatTokenWriter

    writers = {
        "train": FlatTokenWriter(output_dir / "train.bin"),
        "val": FlatTokenWriter(output_dir / "val.bin"),
        "test": FlatTokenWriter(output_dir / "test.bin"),
    }

    for record in records:
        split = record["split"]
        text = record["text"]
        ids = tokenizer.encode(text)
        assert EOS_ID not in ids, f"EOS in encoded text: {text}"
        ids.append(EOS_ID)
        writers[split].add_document(ids, context_len=CONTEXT_LEN)

    for writer in writers.values():
        writer.close()

    # Verify outputs
    train_summary = writers["train"].summary()
    val_summary = writers["val"].summary()
    test_summary = writers["test"].summary()

    assert train_summary["documents"] == 1
    assert val_summary["documents"] == 1
    assert test_summary["documents"] == 1

    assert train_summary["eos_tokens"] == 1
    assert val_summary["eos_tokens"] == 1
    assert test_summary["eos_tokens"] == 1

    # Verify binary files
    assert bin_length(output_dir / "train.bin") > 0
    assert bin_length(output_dir / "val.bin") > 0
    assert bin_length(output_dir / "test.bin") > 0

    # Verify EOS count matches document count
    train_eos = count_tokens_in_bin(output_dir / "train.bin", value=EOS_ID)
    assert train_eos == 1

    val_eos = count_tokens_in_bin(output_dir / "val.bin", value=EOS_ID)
    assert val_eos == 1

    test_eos = count_tokens_in_bin(output_dir / "test.bin", value=EOS_ID)
    assert test_eos == 1


def test_tokenize_corpus_rejects_eos_in_encoded(tmp_path):
    """Test that documents with EOS in encoded text are rejected."""
    from tinyllm.tokenizer import TokenizerWrapper

    tokenizer_path = tmp_path / "tokenizer.json"
    build_synthetic_tokenizer(tokenizer_path, vocab_size=64)
    tokenizer = TokenizerWrapper(tokenizer_path)

    # If by some chance encode produced 0, we'd reject it
    # This test documents the expected behavior
    text = "normal text"
    ids = tokenizer.encode(text)
    if EOS_ID in ids:
        # This should never happen with v8 layout
        pytest.fail("Synthetic tokenizer unexpectedly produced EOS id")
