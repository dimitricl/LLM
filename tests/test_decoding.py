#!/usr/bin/env python3
"""Tests for decoding.py (pure numpy, no MLX)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyllm.decoding import (
    count_ngram_repetitions,
    apply_no_repeat_ngram_mask,
    compute_repetition_penalty,
    detect_looping_output,
    filter_logits_top_k_top_p,
    sample_from_logits,
)


def test_count_ngram_repetitions_basic():
    """Test basic n-gram repetition counting."""
    # No repetition
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    count, is_rep = count_ngram_repetitions(tokens, n=4, threshold=3)
    assert count == 0
    assert is_rep is False

    # With repetition
    tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]  # "1234" repeated 3 times
    count, is_rep = count_ngram_repetitions(tokens, n=4, threshold=3)
    assert count >= 1
    assert is_rep is True


def test_count_ngram_repetitions_short():
    """Test with sequence shorter than n."""
    tokens = [1, 2]
    count, is_rep = count_ngram_repetitions(tokens, n=4, threshold=3)
    assert count == 0
    assert is_rep is False


def test_apply_no_repeat_ngram_mask():
    """Test no-repeat n-gram masking."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    token_ids = [1, 2, 3, 4, 1, 2, 3]  # Last 3 tokens: 1,2,3

    masked = apply_no_repeat_ngram_mask(logits, token_ids, n=4)

    # Token 4 should be masked (would complete "1234" again)
    assert masked[4] == -np.inf
    # Others should be unchanged
    assert masked[0] == 1.0
    assert masked[1] == 2.0


def test_apply_no_repeat_ngram_mask_short():
    """Test no-repeat mask with short history."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    token_ids = [1, 2]  # Too short for n=4

    masked = apply_no_repeat_ngram_mask(logits, token_ids, n=4)
    # Should be unchanged
    np.testing.assert_array_equal(masked, logits)


def test_compute_repetition_penalty():
    """Test repetition penalty."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    token_ids = [1, 3]

    penalized = compute_repetition_penalty(logits, token_ids, penalty=2.0)

    # Tokens 1 and 3 should be halved (2.0/2.0=1.0, 4.0/2.0=2.0)
    assert penalized[1] == 1.0
    assert penalized[3] == 2.0
    # Others unchanged
    assert penalized[0] == 1.0
    assert penalized[2] == 3.0


def test_compute_repetition_penalty_negative():
    """Test repetition penalty on negative logits."""
    logits = np.array([-1.0, -2.0, 3.0, -4.0, 5.0])
    token_ids = [1, 3]

    penalized = compute_repetition_penalty(logits, token_ids, penalty=2.0)

    # Negative logits should be multiplied
    assert penalized[1] == -4.0
    assert penalized[3] == -8.0


def test_detect_looping_output_eos():
    """Test that EOS detection prevents false positive."""
    token_ids = [1, 2, 3, 4, 1, 2, 3, 4, 0]  # Has EOS at end
    is_looping, reason = detect_looping_output(token_ids, eos_id=0, n=4, threshold=3)
    assert is_looping is False
    assert reason == ""


def test_detect_looping_output_no_eos_max_tokens():
    """Test max tokens reached without EOS."""
    token_ids = [1, 2, 3, 4, 5] * 30  # 150 tokens
    is_looping, reason = detect_looping_output(
        token_ids, eos_id=0, n=4, threshold=3, max_tokens=120
    )
    assert is_looping is True
    assert "max_tokens" in reason


def test_detect_looping_output_repetition():
    """Test n-gram repetition detection."""
    token_ids = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]  # Repeated 4-gram
    is_looping, reason = detect_looping_output(token_ids, eos_id=0, n=4, threshold=3)
    assert is_looping is True
    assert "repetition" in reason


def test_detect_looping_output_clean():
    """Test clean output (no looping)."""
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    is_looping, reason = detect_looping_output(token_ids, eos_id=0, n=4, threshold=3)
    assert is_looping is False
    assert reason == ""


def test_filter_logits_top_k():
    """Test top-k filtering."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = filter_logits_top_k_top_p(logits, top_k=2)

    # Only top 2 should remain
    non_inf = filtered[filtered > -np.inf]
    assert len(non_inf) == 2
    # They should be the highest values
    assert 5.0 in non_inf
    assert 4.0 in non_inf


def test_filter_logits_top_p():
    """Test top-p filtering."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = filter_logits_top_k_top_p(logits, top_p=0.5)

    # Cumulative prob of top tokens should be >= 0.5
    probs = np.exp(filtered - np.max(filtered))
    probs = probs / np.sum(probs)
    cumulative = np.cumsum(np.sort(probs)[::-1])
    assert cumulative[0] >= 0.5  # Top token alone


def test_filter_logits_both():
    """Test combined top-k and top-p."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = filter_logits_top_k_top_p(logits, top_k=3, top_p=0.9)

    # Should respect both constraints
    non_inf = filtered[filtered > -np.inf]
    assert len(non_inf) <= 3  # top-k constraint


def test_sample_from_logits_greedy():
    """Test greedy sampling (temperature=0)."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    token = sample_from_logits(logits, temperature=0, seed=42)
    assert token == 4  # Argmax


def test_sample_from_logits_temperature():
    """Test temperature sampling."""
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    token1 = sample_from_logits(logits, temperature=1.0, seed=42)
    token2 = sample_from_logits(logits, temperature=1.0, seed=43)

    # Different seeds should give different results (probabilistic)
    # Though might occasionally be same
    # We just verify it returns a valid token
    assert 0 <= token1 < len(logits)
    assert 0 <= token2 < len(logits)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
