import mlx.core as mx

from tinyllm.model import RMSNorm, apply_rope, rope_tables


def test_rms_norm_has_unit_rms():
    norm = RMSNorm(64)
    x = mx.random.normal((2, 5, 64)) * 3.0
    out = norm(x)
    rms = mx.sqrt(mx.mean(out * out, axis=-1))
    assert mx.allclose(rms, mx.ones_like(rms), atol=1e-4)


def test_rope_at_position_zero_is_identity():
    x = mx.random.normal((1, 2, 1, 16))
    cos, sin = rope_tables(seq_len=4, head_dim=16, theta=10_000.0)
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
