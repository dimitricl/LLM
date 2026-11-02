import mlx.core as mx
import numpy as np

from tinyllm.config import ModelConfig
from tinyllm.model import Attention, RMSNorm, apply_rope, causal_mask, rope_tables


def test_rms_norm_has_unit_rms():
    norm = RMSNorm(64)
    x = mx.random.normal((2, 5, 64)) * 3.0
    out = norm(x)
    rms = mx.sqrt(mx.mean(out * out, axis=-1))
    assert mx.allclose(rms, mx.ones_like(rms), atol=1e-4)


def test_rope_at_position_zero_is_identity():
    x = mx.random.normal((1, 2, 1, 16))
    cos, sin = rope_tables(seq_len=4, head_dim=16, theta=10_000.0)
    # Apply rope with 4D tensor shape (batch, heads, seq, head_dim)
    assert mx.allclose(apply_rope(x, cos, sin, offset=0), x, atol=1e-6)


def test_rope_preserves_norm():
    x = mx.random.normal((1, 2, 5, 16))
    cos, sin = rope_tables(seq_len=8, head_dim=16, theta=10_000.0)
    out = apply_rope(x, cos, sin, offset=0)
    assert mx.allclose(
        mx.linalg.norm(x, axis=-1), mx.linalg.norm(out, axis=-1), atol=1e-4
    )


def test_rope_depends_only_on_relative_position():
    q = mx.random.normal((1, 1, 1, 16))
    k = mx.random.normal((1, 1, 1, 16))
    cos, sin = rope_tables(seq_len=12, head_dim=16, theta=10_000.0)
    q3 = apply_rope(q, cos, sin, offset=3)
    k5 = apply_rope(k, cos, sin, offset=5)
    q0 = apply_rope(q, cos, sin, offset=0)
    k2 = apply_rope(k, cos, sin, offset=2)
    assert mx.allclose(mx.sum(q3 * k5), mx.sum(q0 * k2), atol=1e-4)


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=320,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        context_len=32,
    )


def test_causal_mask_lower_triangle():
    mask = np.array(causal_mask(3, 3))
    assert np.all(np.isfinite(mask[np.tril_indices(3)]))
    assert np.all(np.isneginf(mask[np.triu_indices(3, k=1)]))


def test_causal_mask_single_query_sees_everything():
    assert np.all(np.isfinite(np.array(causal_mask(1, 5))))


def test_attention_output_shape():
    cfg = _cfg()
    attention = Attention(cfg)
    x = mx.random.normal((2, 16, cfg.d_model))
    cos, sin = rope_tables(16, cfg.head_dim, cfg.rope_theta)
    out = attention(x, cos, sin)
    assert out.shape == (2, 16, cfg.d_model)


def test_attention_is_causal():
    cfg = _cfg()
    attention = Attention(cfg)
    mx.random.seed(0)
    x = mx.random.normal((1, 10, cfg.d_model))
    cos, sin = rope_tables(10, cfg.head_dim, cfg.rope_theta)
    out_a = attention(x, cos, sin)
    mask = (mx.arange(10) == 5)[None, :, None]
    changed = mx.where(mask, x + 1.0, x)
    out_b = attention(changed, cos, sin)
    assert mx.allclose(out_a[:, :5], out_b[:, :5], atol=1e-5)
    assert not mx.allclose(out_a[:, 5:], out_b[:, 5:], atol=1e-5)
