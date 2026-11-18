"""Tests for v8 pre-training corpus pipeline.

These tests use synthetic data and a tiny tokenizer to verify the logic
without requiring the full tokenizer.json file or MLX.
"""
from pathlib import Path
import json
import numpy as np
import pytest

from tinyllm.pretrain_v8 import (
    EOS_ID,
    PAD_ID,
    CONTEXT_LEN,
    FlatTokenWriter,
    alpha_ratio,
    language_ok,
    language_score,
    normalize_text,
    sha1_digest,
    split_for_text,
    word_tokens,
    repetition_ratio,
    shingle_hashes_from_words,
    sample_words,
    make_minhash_params,
    minhash_signature,
    band_keys,
    jaccard_estimate,
    LeakIndex,
    bin_length,
    count_tokens_in_bin,
    json_dump,
    counter_dict,
    estimated_seconds,
)


def test_normalize_text_keeps_content():
    text = "  Bonjour  \n\n  monde  \n"
    normalized = normalize_text(text, "text")
    assert normalized == "Bonjour\n\nmonde"


def test_normalize_code_preserves_structure():
    code = "def f():\n    return 1\n\n\ndef g():\n    return 2"
    normalized = normalize_text(code, "code")
    assert "def f():" in normalized
    assert "\n\n" in normalized
    # Code normalization collapses 4+ newlines to 3, not 2
    assert normalized.count("\n\n\n") <= 1


def test_word_tokens_extracts_words():
    text = "Bonjour, le monde!"
    words = word_tokens(text)
    assert words == ["bonjour", "le", "monde"]


def test_alpha_ratio_ignores_spaces():
    text = "abc def ghi"
    assert alpha_ratio(text) == 1.0
    text = "abc 123 def"
    assert alpha_ratio(text) == 0.6666666666666666


def test_language_score_french():
    text = "Le chat mange la souris. Il est content."
    score = language_score(text, "fr")
    assert score > 0.1


def test_language_score_english():
    text = "The cat eats the mouse. He is happy."
    score = language_score(text, "en")
    assert score > 0.1


def test_language_ok_threshold():
    text = "Le chat mange la souris."
    ok, score = language_ok(text, "fr", 0.01)
    assert ok
    assert score > 0.01


def test_repetition_ratio_simple():
    words = ["a", "b", "c", "a", "b", "c"] * 10
    ratio, _ = repetition_ratio(words, n=3, max_words=100)
    assert ratio > 0.5


def test_repetition_ratio_unique():
    words = list("abcdefghijklmnopqrstuvwxyz")
    ratio, _ = repetition_ratio(words, n=3, max_words=100)
    assert ratio == 0.0


def test_sample_words():
    words = list(range(100))
    sampled = sample_words(words, 10)
    assert len(sampled) == 10
    assert sampled[0] == words[0]
    # sample_words uses step-based sampling, so last element is near the end
    assert sampled[-1] >= words[-10]


def test_shingle_hashes_are_deterministic():
    words = ["a", "b", "c", "d"]
    hashes1 = shingle_hashes_from_words(words, 2)
    hashes2 = shingle_hashes_from_words(words, 2)
    assert np.array_equal(hashes1, hashes2)


def test_minhash_signature_length():
    params = make_minhash_params(num_hashes=12, seed=42)
    hashes = shingle_hashes_from_words(["a", "b", "c"], 2)
    signature = minhash_signature(hashes, params)
    assert len(signature) == 12


def test_band_keys():
    signature = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    keys = band_keys(signature, rows=3)
    assert len(keys) == 4
    assert keys[0] == (1, 2, 3)


def test_jaccard_estimate():
    left = np.array([1, 2, 3, 4])
    right = np.array([1, 2, 3, 4])
    assert jaccard_estimate(left, right) == 1.0
    right = np.array([5, 6, 7, 8])
    assert jaccard_estimate(left, right) == 0.0


def test_sha1_digest_is_deterministic():
    text = "Hello, world!"
    digest1 = sha1_digest(text)
    digest2 = sha1_digest(text)
    assert digest1 == digest2
    assert len(digest1) == 20


def test_split_for_text_distribution():
    from tinyllm.pretrain_v8 import VAL_FRAC, TEST_FRAC
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(1000):
        text = f"Document {i}"
        split = split_for_text(text, val_frac=VAL_FRAC, test_frac=TEST_FRAC)
        counts[split] += 1
    assert counts["train"] > 900
    assert counts["val"] > 5
    assert counts["test"] > 5


def test_leak_index_exact_match():
    records = [{"text": "Le chat mange la souris."}]
    leak_index = LeakIndex(records, num_hashes=4, bands=2, seed=42)
    digest = sha1_digest("Le chat mange la souris.")
    words = word_tokens("Le chat mange la souris.")
    params = make_minhash_params(4, 42)
    signature = minhash_signature(
        shingle_hashes_from_words(words, 5), params
    )
    reason = leak_index.check(digest, words, signature)
    assert reason == "leak_exact"


def test_leak_index_no_match():
    # Use longer, more distinctive texts to avoid false positives
    records = [{"text": "The quick brown fox jumps over the lazy dog repeatedly."}]
    leak_index = LeakIndex(records, num_hashes=12, bands=4, seed=42)
    digest = sha1_digest("A completely different sentence about other topics.")
    words = word_tokens("A completely different sentence about other topics.")
    params = make_minhash_params(12, 42)
    signature = minhash_signature(
        shingle_hashes_from_words(words, 5), params
    )
    reason = leak_index.check(digest, words, signature)
    assert reason is None


def test_flat_token_writer_validates_eos(tmp_path):
    path = tmp_path / "test.bin"
    writer = FlatTokenWriter(path)
    with pytest.raises(ValueError, match="must end with exactly one EOS id"):
        writer.add_document([1, 2, 3], context_len=CONTEXT_LEN)


def test_flat_token_writer_rejects_internal_eos(tmp_path):
    path = tmp_path / "test.bin"
    writer = FlatTokenWriter(path)
    with pytest.raises(ValueError, match="EOS id found inside"):
        writer.add_document([1, 2, EOS_ID, 3, EOS_ID], context_len=CONTEXT_LEN)


def test_flat_token_writer_valid(tmp_path):
    path = tmp_path / "test.bin"
    writer = FlatTokenWriter(path)
    writer.add_document([1, 2, 3, EOS_ID], context_len=CONTEXT_LEN)
    writer.add_document([4, 5, EOS_ID], context_len=CONTEXT_LEN)
    writer.close()

    summary = writer.summary(context_len=CONTEXT_LEN)
    assert summary["documents"] == 2
    assert summary["tokens"] == 7  # 4 + 3 tokens
    assert summary["eos_tokens"] == 2
    assert summary["mean_tokens_per_document"] == 3.5


def test_flat_token_writer_counts_longer_than_context(tmp_path):
    path = tmp_path / "test.bin"
    writer = FlatTokenWriter(path)
    # Use non-zero ids to avoid EOS_ID (0) in the middle
    long_doc = list(range(1, 601)) + [EOS_ID]
    writer.add_document(long_doc, context_len=CONTEXT_LEN)
    writer.close()

    summary = writer.summary(context_len=CONTEXT_LEN)
    assert summary["documents_longer_than_context"] == 1
    assert summary["documents_longer_than_context_pct"] == 100.0


def test_bin_length(tmp_path):
    # Test valid file
    path = tmp_path / "test.bin"
    data = np.array([1, 2, 3], dtype=np.uint16)
    data.tofile(path)
    assert bin_length(path) == 3

    # Test non-existent file
    assert bin_length("/nonexistent") == 0


def test_count_tokens_in_bin(tmp_path):
    path = tmp_path / "test.bin"
    data = np.array([1, 2, EOS_ID, 4, 5, EOS_ID], dtype=np.uint16)
    data.tofile(path)
    count = count_tokens_in_bin(path, value=EOS_ID)
    assert count == 2


def test_json_dump_atomic(tmp_path):
    path = tmp_path / "test.json"
    payload = {"key": "value"}
    json_dump(path, payload)
    assert path.exists()
    assert json.loads(path.read_text()) == payload


def test_counter_dict():
    from collections import Counter
    counter = Counter(["a", "b", "a", "c", "a"])
    result = counter_dict(counter)
    assert result == {"a": 3, "b": 1, "c": 1}


def test_estimated_seconds():
    assert estimated_seconds(100_000, 1000) == 100.0
    assert estimated_seconds(100_000, 0) == float("inf")


def test_eos_id_is_zero():
    assert EOS_ID == 0


def test_pad_id_is_one():
    assert PAD_ID == 1


def test_context_len_is_512():
    assert CONTEXT_LEN == 512
