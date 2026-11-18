import mlx.core as mx
import numpy as np

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.sample import (
    apply_repetition_penalty,
    filter_logits,
    generate_ids,
    sample_token,
)


def _model() -> TinyLM:
    mx.random.seed(0)
    return TinyLM(
        ModelConfig(
            vocab_size=320, d_model=32, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_hidden=64, context_len=16,
        )
    )


def test_generate_ids_length_without_eos():
    ids = generate_ids(
        _model(), [1, 2, 3], max_tokens=5, temperature=0.8, stop_at_eos=False, seed=0
    )
    assert len(ids) == 8


def test_greedy_generation_is_deterministic():
    a = generate_ids(_model(), [1, 2], max_tokens=6, temperature=0.0, stop_at_eos=False)
    b = generate_ids(_model(), [1, 2], max_tokens=6, temperature=0.0, stop_at_eos=False)
    assert a == b


def test_top_k_keeps_single_token():
    logits = mx.array([1.0, 4.0, 2.0, 3.0])
    filtered = filter_logits(logits, top_k=1)
    assert int(mx.argmax(filtered)) == 1
    assert np.isneginf(np.array(filtered)).sum() == 3


def test_top_p_keeps_at_least_one_token():
    logits = mx.array([10.0, 0.0, 0.0])
    filtered = filter_logits(logits, top_p=0.1)
    assert np.isfinite(np.array(filtered)).sum() >= 1


def test_repetition_penalty_changes_seen_token():
    logits = mx.array([0.0, 0.0, 3.0, 0.0])
    penalized = apply_repetition_penalty(logits, [2, 2], penalty=2.0)
    assert abs(float(penalized[2]) - 1.5) < 1e-6


def test_sample_token_greedy_matches_argmax():
    logits = mx.array([0.0, 5.0, 1.0])
    assert sample_token(logits, temperature=0.0, key=mx.random.key(0)) == 1
