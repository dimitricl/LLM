#!/usr/bin/env python3
"""Tests for build_sft_v8.py"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_sft_v8 import (
    convert_ultrachat,
    convert_openhermes,
    convert_alpaca,
    convert_openorca,
    convert_to_messages,
    convert_messages,
    compute_ngram_repetition_ratio,
    has_final_punctuation,
    has_latex,
    has_long_code,
    is_factual_outlier,
    detect_language,
    compute_content_hash,
    generate_refusal_examples,
    generate_in_scope_examples,
)


def test_convert_ultrachat():
    """Test UltraChat conversion."""
    record = {
        "id": "1",
        "data": ["Hello", "Hi there", "How are you?", "I'm good"]
    }
    messages = convert_ultrachat(record)
    assert messages is not None
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_convert_openhermes():
    """Test OpenHermes conversion."""
    record = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"}
        ]
    }
    messages = convert_openhermes(record)
    assert messages is not None
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_convert_alpaca():
    """Test Alpaca conversion."""
    record = {
        "instruction": "Say hello",
        "input": "",
        "output": "Hello!"
    }
    messages = convert_alpaca(record)
    assert messages is not None
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_convert_openorca():
    """Test OpenOrca conversion."""
    record = {
        "question": "What is 2+2?",
        "response": "4"
    }
    messages = convert_openorca(record)
    assert messages is not None
    assert len(messages) == 2


def test_convert_messages():
    """Raw files that already use {"messages": [...]} must be accepted."""
    record = {
        "messages": [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour !"},
        ]
    }
    messages = convert_messages(record)
    assert messages is not None
    assert len(messages) == 2

    # The shape-based dispatcher must find it without a filename hint.
    messages = convert_to_messages(record, "general_fr")
    assert messages is not None
    assert messages[0]["role"] == "user"


def test_detect_language_realistic():
    """Require a real FR/EN margin, not substring matches."""
    fr_text = (
        "Je pense que la voiture est dans le garage et que nous devons "
        "partir avec elle pour aller au marché."
    )
    en_text = (
        "I think that the car is in the garage and that we should leave "
        "with it to go to the market."
    )
    assert detect_language(fr_text) == "fr"
    assert detect_language(en_text) == "en"


def test_convert_to_messages():
    """Test generic conversion."""
    # UltraChat
    record = {"data": ["Q", "A"]}
    messages = convert_to_messages(record, "ultrachat")
    assert messages is not None

    # OpenHermes
    record = {"conversations": [{"from": "human", "value": "Q"}]}
    messages = convert_to_messages(record, "openhermes")
    # Need full conversation
    record = {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}
    messages = convert_to_messages(record, "openhermes")
    assert messages is not None

    # Alpaca
    record = {"instruction": "Q", "output": "A"}
    messages = convert_to_messages(record, "alpaca")
    assert messages is not None

    # Unknown format
    messages = convert_to_messages({}, "unknown")
    assert messages is None


def test_compute_ngram_repetition_ratio():
    """Test n-gram repetition detection."""
    # No repetition
    text = "a b c d e f g h i j"
    ratio = compute_ngram_repetition_ratio(text, n=4)
    assert ratio == 0.0

    # With repetition
    text = "a b c d a b c d a b c d"
    ratio = compute_ngram_repetition_ratio(text, n=4)
    assert ratio > 0.0

    # Short text
    text = "a b"
    ratio = compute_ngram_repetition_ratio(text, n=4)
    assert ratio == 0.0


def test_has_final_punctuation():
    """Test final punctuation detection."""
    assert has_final_punctuation("Hello.") is True
    assert has_final_punctuation("Hello!") is True
    assert has_final_punctuation("Hello?") is True
    assert has_final_punctuation("Hello") is False
    assert has_final_punctuation("") is False


def test_has_latex():
    """Test LaTeX detection."""
    assert has_latex("This is $x^2$") is True
    assert has_latex("Use \\frac{1}{2}") is True
    assert has_latex("No math here") is False


def test_has_long_code():
    """Test long code detection."""
    short_code = "```python\nprint('hi')\n```"
    assert has_long_code(short_code) is False

    long_code = "```python\n" + "\n".join([f"line {i}" for i in range(15)]) + "\n```"
    assert has_long_code(long_code) is True


def test_is_factual_outlier():
    """Test factual outlier detection."""
    normal = "Paris is the capital of France."
    assert is_factual_outlier(normal) is False

    many_dates = "On 01/01/2020, then 02/02/2020, then 03/03/2020, then 04/04/2020, then 05/05/2020, then 06/06/2020."
    assert is_factual_outlier(many_dates) is True


def test_detect_language():
    """Test language detection."""
    fr_text = "Le chat est sur la table."
    assert detect_language(fr_text) == "fr"

    en_text = "The cat is on the table."
    assert detect_language(en_text) == "en"

    mixed = "Le the cat est on table."
    # Should still pick one
    lang = detect_language(mixed)
    assert lang in ("fr", "en", "unknown")


def test_compute_content_hash():
    """Test content hashing."""
    messages1 = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    messages2 = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    messages3 = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Bye"}]

    h1 = compute_content_hash(messages1)
    h2 = compute_content_hash(messages2)
    h3 = compute_content_hash(messages3)

    assert h1 == h2  # Same content
    assert h1 != h3  # Different content


def test_generate_refusal_examples():
    """Test refusal example generation."""
    examples = generate_refusal_examples(5)
    assert len(examples) == 5
    for ex in examples:
        assert len(ex) == 2
        assert ex[0]["role"] == "user"
        assert ex[1]["role"] == "assistant"
        # Check for refusal phrases (more flexible matching)
        content_lower = ex[1]["content"].lower()
        assert "ne sais" in content_lower or "ne peux" in content_lower or "désolé" in content_lower


def test_generate_in_scope_examples():
    """Test in-scope example generation."""
    examples = generate_in_scope_examples(5)
    assert len(examples) == 5
    for ex in examples:
        assert len(ex) == 2
        assert ex[0]["role"] == "user"
        assert ex[1]["role"] == "assistant"
        # Should NOT be refusals
        assert not any(phrase in ex[1]["content"] for phrase in ["Je ne sais pas", "Je ne peux pas"])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
