#!/usr/bin/env python3
"""Tests for eval_quality.py (metric computation logic)"""

import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_quality import (
    load_lexicon,
    load_prompts,
    compute_distinct,
    compute_oov_rate,
    check_keyword_match,
    is_refusal,
    evaluate_outputs,
    compute_confidence_interval,
    format_ci,
)


def test_load_lexicon():
    """Test lexicon loading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# Comment\n")
        f.write("bonjour\n")
        f.write("salut\n")
        f.write("monsieur\n")
        path = Path(f.name)

    try:
        lexicon = load_lexicon(path)
        assert "bonjour" in lexicon
        assert "salut" in lexicon
        assert "monsieur" in lexicon
        assert len(lexicon) == 3
    finally:
        path.unlink()


def test_load_prompts():
    """Test prompts loading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "category": "greeting", "language": "fr", "prompt": "Bonjour"}\n')
        f.write('{"id": "2", "category": "factual", "language": "fr", "prompt": "Paris?", "expected_keywords": ["Paris"]}\n')
        path = Path(f.name)

    try:
        prompts = load_prompts(path)
        assert len(prompts) == 2
        assert prompts[0]["category"] == "greeting"
        assert prompts[1]["expected_keywords"] == ["Paris"]
    finally:
        path.unlink()


def test_compute_distinct():
    """Test distinct-n computation."""
    # All unique
    text = "a b c d e"
    assert compute_distinct(text, 1) == 1.0
    assert compute_distinct(text, 2) == 1.0

    # With repetition
    text = "a a b b c"
    assert compute_distinct(text, 1) == 0.6  # 3 unique / 5 total
    # Bigrams: "a a", "a b", "b b", "b c" - all unique
    assert compute_distinct(text, 2) == 1.0

    # With actual bigram repetition
    text = "a b a b c"
    assert compute_distinct(text, 1) == 0.6  # 3 unique / 5 total
    # Bigrams: "a b", "b a", "a b", "b c" - "a b" repeated
    assert compute_distinct(text, 2) == 0.75  # 3 unique / 4 total

    # Short text
    text = "a b"
    assert compute_distinct(text, 3) == 0.0


def test_compute_oov_rate():
    """Test OOV rate computation."""
    lexicon = {"bonjour", "salut", "monsieur", "madame"}

    # All in lexicon
    text = "bonjour salut monsieur"
    assert compute_oov_rate(text, lexicon) == 0.0

    # Half OOV
    text = "bonjour salut xyz abc"
    assert compute_oov_rate(text, lexicon) == 0.5

    # All OOV
    text = "xyz abc def"
    assert compute_oov_rate(text, lexicon) == 1.0


def test_check_keyword_match():
    """Test keyword matching."""
    text = "La capitale de la France est Paris"

    assert check_keyword_match(text, ["Paris"]) is True
    assert check_keyword_match(text, ["France"]) is True
    assert check_keyword_match(text, ["Paris", "France"]) is True
    assert check_keyword_match(text, ["Londres"]) is False
    assert check_keyword_match(text, ["paris"]) is True  # Case-insensitive


def test_is_refusal():
    """Test refusal detection."""
    assert is_refusal("Je ne sais pas.") is True
    assert is_refusal("Je ne peux pas répondre.") is True
    assert is_refusal("Désolé, je ne dispose pas de cette information.") is True
    assert is_refusal("I don't know") is True
    assert is_refusal("I cannot answer") is True
    assert is_refusal("La réponse est Paris.") is False
    assert is_refusal("Oui, bien sûr.") is False


def test_evaluate_outputs_basic():
    """Test basic evaluation."""
    lexicon = {"bonjour", "salut", "monsieur", "madame", "paris", "france"}

    outputs = [
        {
            "response": "Bonjour monsieur",
            "token_ids": [1, 2, 3, 4, 0],  # Has EOS
            "eos_id": 0,
            "max_tokens": 120,
            "category": "greeting",
        },
        {
            "response": "Paris est la capitale",
            "token_ids": [5, 6, 7, 8, 9, 0],
            "eos_id": 0,
            "max_tokens": 120,
            "category": "factual",
            "expected_keywords": ["Paris"],
        },
    ]

    metrics = evaluate_outputs(outputs, lexicon)

    assert metrics["total"] == 2
    assert metrics["loop_rate"] == 0.0
    assert metrics["distinct_1"] > 0.5
    assert metrics["factual_accuracy"] == 1.0


def test_evaluate_outputs_looping():
    """Test loop detection."""
    lexicon = set()

    outputs = [
        {
            "response": "a b c d a b c d a b c d",
            "token_ids": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],  # Repeated 4-gram
            "eos_id": 0,
            "max_tokens": 120,
            "category": "greeting",
        },
    ]

    metrics = evaluate_outputs(outputs, lexicon)
    assert metrics["loop_rate"] == 1.0


def test_evaluate_outputs_no_eos():
    """Test no EOS detection."""
    lexicon = set()

    outputs = [
        {
            "response": "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "token_ids": list(range(1, 27)),  # No EOS (id 0)
            "eos_id": 0,
            "max_tokens": 26,
            "category": "greeting",
        },
    ]

    metrics = evaluate_outputs(outputs, lexicon)
    assert metrics["max_tokens_rate"] == 1.0


def test_evaluate_outputs_refusals():
    """Test refusal metrics."""
    lexicon = set()

    outputs = [
        {
            "response": "Je ne sais pas.",
            "token_ids": [1, 2, 3, 4, 0],
            "eos_id": 0,
            "max_tokens": 120,
            "category": "out_of_scope",
        },
        {
            "response": "Paris",
            "token_ids": [5, 0],
            "eos_id": 0,
            "max_tokens": 120,
            "category": "factual",
        },
        {
            "response": "Je ne peux pas répondre.",
            "token_ids": [6, 7, 8, 9, 0],
            "eos_id": 0,
            "max_tokens": 120,
            "category": "greeting",  # False refusal
        },
    ]

    metrics = evaluate_outputs(outputs, lexicon)

    assert metrics["refusal_correct"] == 1.0  # Correct refusal on out-of-scope
    # 2 in-scope prompts (factual + greeting), 1 is a refusal, so 0.5
    assert metrics["false_refusal"] == 0.5


def test_compute_confidence_interval():
    """Test confidence interval computation."""
    # Perfect match
    lower, upper = compute_confidence_interval(1.0, 100)
    assert 0.9 < lower < 1.0
    assert upper >= 0.99  # Very close to 1.0

    # 50% match
    lower, upper = compute_confidence_interval(0.5, 100)
    assert 0.3 < lower < 0.5
    assert 0.5 < upper < 0.7

    # Small sample
    lower, upper = compute_confidence_interval(0.5, 10)
    assert lower < upper


def test_format_ci():
    """Test CI formatting."""
    result = format_ci(0.75, 100)
    assert "±" in result
    assert "0.75" in result

    result = format_ci(0.5, 0)
    assert "N/A" in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
