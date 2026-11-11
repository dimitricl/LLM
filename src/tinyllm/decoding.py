"""
Pure numpy functions for decoding penalties and n-gram repetition detection.
Testable without MLX.
"""

import numpy as np
from typing import List, Tuple, Optional


def count_ngram_repetitions(
    token_ids: List[int],
    n: int = 4,
    threshold: int = 3,
) -> Tuple[int, bool]:
    """
    Count n-gram repetitions and check if threshold is exceeded.

    Args:
        token_ids: List of token IDs
        n: N-gram size (default 4)
        threshold: Minimum count to flag as repetition (default 3)

    Returns:
        (count_exceeded, is_repetitive) where:
        - count_exceeded: number of n-grams with count >= threshold
        - is_repetitive: True if any n-gram exceeds threshold
    """
    if len(token_ids) < n:
        return 0, False

    # Build n-grams
    ngrams = []
    for i in range(len(token_ids) - n + 1):
        ngram = tuple(token_ids[i:i+n])
        ngrams.append(ngram)

    # Count occurrences
    from collections import Counter
    ngram_counts = Counter(ngrams)

    # Count n-grams exceeding threshold
    count_exceeded = sum(1 for count in ngram_counts.values() if count >= threshold)
    is_repetitive = count_exceeded > 0

    return count_exceeded, is_repetitive


def apply_no_repeat_ngram_mask(
    logits: np.ndarray,
    token_ids: List[int],
    n: int = 4,
) -> np.ndarray:
    """
    Apply no-repeat n-gram penalty by setting banned tokens to -inf.

    If the last n-1 tokens match a previous n-gram, the next token
    that would complete the repetition is masked out.

    Args:
        logits: Logits array of shape (vocab_size,)
        token_ids: List of generated token IDs so far
        n: N-gram size (default 4)

    Returns:
        Modified logits with banned tokens set to -inf
    """
    if len(token_ids) < n - 1:
        return logits

    vocab_size = logits.shape[0]
    banned_tokens = set()

    # Get the last n-1 tokens
    last_n_minus_1 = tuple(token_ids[-(n-1):])

    # Scan previous n-grams
    for i in range(len(token_ids) - n + 1):
        prev_ngram = tuple(token_ids[i:i+n-1])
        if prev_ngram == last_n_minus_1:
            # Would repeat: ban the next token
            if i + n - 1 < len(token_ids):
                banned_token = token_ids[i + n - 1]
                banned_tokens.add(banned_token)

    # Apply mask
    logits = logits.copy()
    for token in banned_tokens:
        if 0 <= token < vocab_size:
            logits[token] = -np.inf

    return logits


def compute_repetition_penalty(
    logits: np.ndarray,
    token_ids: List[int],
    penalty: float = 1.15,
) -> np.ndarray:
    """
    Apply repetition penalty to tokens that have appeared.

    Args:
        logits: Logits array of shape (vocab_size,)
        token_ids: List of generated token IDs
        penalty: Penalty factor > 1.0 (default 1.15)

    Returns:
        Modified logits
    """
    if penalty <= 1.0 or not token_ids:
        return logits

    logits = logits.copy()
    seen_tokens = set(token_ids)

    for token in seen_tokens:
        if 0 <= token < logits.shape[0]:
            if logits[token] > 0:
                logits[token] /= penalty
            else:
                logits[token] *= penalty

    return logits


def detect_looping_output(
    token_ids: List[int],
    eos_id: int,
    n: int = 4,
    threshold: int = 3,
    max_tokens: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Detect if output is looping or failing to stop.

    Args:
        token_ids: Generated token IDs
        eos_id: End-of-sequence token ID
        n: N-gram size for repetition check
        threshold: Repetition threshold
        max_tokens: If set, check if max tokens reached without EOS

    Returns:
        (is_looping, reason) where reason is a string explaining why
    """
    # Check for EOS
    if eos_id in token_ids:
        return False, ""

    # Check if max tokens reached without EOS
    if max_tokens is not None and len(token_ids) >= max_tokens:
        return True, f"max_tokens ({max_tokens}) reached without EOS"

    # Check n-gram repetition
    _, is_repetitive = count_ngram_repetitions(token_ids, n=n, threshold=threshold)
    if is_repetitive:
        return True, f"{n}-gram repetition (threshold {threshold})"

    return False, ""


def filter_logits_top_k_top_p(
    logits: np.ndarray,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> np.ndarray:
    """
    Apply top-k and top-p filtering to logits.

    Args:
        logits: Logits array of shape (vocab_size,)
        top_k: Keep top k tokens (None = disabled)
        top_p: Keep tokens with cumulative probability >= top_p (None = disabled)

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    logits = logits.copy()

    if top_k is not None and top_k > 0:
        keep = min(top_k, logits.shape[0])
        cutoff = np.sort(logits)[-keep]
        logits = np.where(logits < cutoff, -np.inf, logits)

    if top_p is not None and top_p < 1.0:
        # Softmax to get probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # Sort by probability
        order = np.argsort(probs)[::-1]
        ordered_probs = probs[order]

        # Find cutoff for cumulative probability
        cumulative = np.cumsum(ordered_probs)
        # Keep tokens where cumulative <= top_p
        # But always keep at least the first token
        mask = cumulative <= top_p
        mask[0] = True  # Always keep highest prob

        # Apply mask
        logits = np.where(mask[order], logits, -np.inf)

    return logits


def sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    seed: int = 0,
) -> int:
    """
    Sample a token from logits with temperature.

    Args:
        logits: Logits array
        temperature: Sampling temperature (0 = greedy)
        seed: Random seed

    Returns:
        Sampled token ID
    """
    if temperature <= 0:
        return int(np.argmax(logits))

    rng = np.random.RandomState(seed)
    probs = np.exp(logits / temperature - np.max(logits / temperature))
    probs = probs / np.sum(probs)
    return int(rng.choice(len(probs), p=probs))
