import mlx.core as mx
import mlx.nn as nn

from tinyllm.config import ModelConfig


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
    cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    return cos, sin


def apply_rope(
    x: mx.array, cos: mx.array, sin: mx.array, offset: int = 0
) -> mx.array:
    seq_len = x.shape[-2]
    cos = cos[offset : offset + seq_len]
    sin = sin[offset : offset + seq_len]
    if x.ndim == 4:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def causal_mask(q_len: int, kv_len: int, dtype=mx.float32) -> mx.array:
    q_pos = mx.arange(q_len)[:, None] + (kv_len - q_len)
    k_pos = mx.arange(kv_len)[None, :]
    return mx.where(k_pos <= q_pos, 0.0, -mx.inf).astype(dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache=None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = 0 if cache is None else cache.length
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)
        if cache is not None:
            k, v = cache.update(k, v)

        repeats = self.n_heads // self.n_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + causal_mask(seq_len, k.shape[2], scores.dtype)
        probs = mx.softmax(scores, axis=-1)
        out = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out)
