import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.weight


def rope_tables(seq_len: int, head_dim: int, theta: float = 10_000.0) -> tuple[mx.array, mx.array]:
    positions = mx.arange(seq_len)
    freqs = mx.exp(
        mx.arange(0, head_dim, 2, dtype=mx.float32)
        * (-mx.log(theta) / head_dim)
    )
    angles = positions[:, None] * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def apply_rope(
    x: mx.array, cos: mx.array, sin: mx.array, offset: int = 0
) -> mx.array:
    seq_len = x.shape[-2]
    cos = cos[offset : offset + seq_len]
    sin = sin[offset : offset + seq_len]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = mx.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)
