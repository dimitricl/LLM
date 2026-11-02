import mlx.core as mx
import mlx.nn as nn

from tinyllm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def rope_tables(
    seq_len: int,
    head_dim: int,
    theta: float = 10_000.0,
    dtype=mx.float32,
) -> tuple[mx.array, mx.array]:
    positions = mx.arange(seq_len)
    freqs = mx.exp(
        mx.arange(0, head_dim, 2, dtype=mx.float32)
        * (-mx.log(theta) / head_dim)
    )
    angles = positions[:, None] * freqs[None, :]
    cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    return cos.astype(dtype), sin.astype(dtype)


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


class KVCache:
    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


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

        mask = "causal" if seq_len > 1 else None
        # Causal masking assumes a prefill starts with an empty cache.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out)

    def attention_reference(
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


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = MLP(cfg)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, cache=None
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.ffn(self.ffn_norm(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def __call__(self, tokens: mx.array, cache=None) -> mx.array:
        x = self.tok_emb(tokens)
        cos, sin = rope_tables(
            self.cfg.context_len,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            dtype=x.dtype,
        )
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            x = layer(x, cos, sin, layer_cache)
        x = self.norm(x)
        return (x @ self.tok_emb.weight.T).astype(mx.float32)

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(parameter.size for _, parameter in tree_flatten(self.parameters()))

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]
