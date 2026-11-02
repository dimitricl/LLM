import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.quantize import load_model, quantize_linears, save_quantized
from tinyllm.tokenizer import train_tokenizer


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=320, d_model=32, n_layers=1, n_heads=4,
        n_kv_heads=2, ffn_hidden=64, context_len=16,
    )


def _model() -> TinyLM:
    mx.random.seed(0)
    return TinyLM(_cfg())


def test_quantize_preserves_shape_and_finiteness():
    model = quantize_linears(_model(), group_size=32, bits=4)
    out = model(mx.array([[1, 2, 3, 4]]))
    assert out.shape == (1, 4, 320)
    assert bool(mx.all(mx.isfinite(out)))


def test_quantized_model_is_smaller():
    model = _model()
    fp32_bytes = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    quantized = quantize_linears(model, group_size=32, bits=4)
    q_bytes = sum(p.nbytes for _, p in tree_flatten(quantized.parameters()))
    assert q_bytes < fp32_bytes * 0.65


def test_save_load_roundtrip(tmp_path):
    model = quantize_linears(_model())
    tokens = mx.array([[1, 2, 3, 4]])
    expected = np.array(model(tokens))

    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(["hello world " * 50], vocab_size=320, out_path=tokenizer_path)
    out_dir = tmp_path / "model"
    save_quantized(model, _cfg(), tokenizer_path, out_dir)

    loaded, tokenizer = load_model(out_dir)
    np.testing.assert_allclose(np.array(loaded(tokens)), expected, atol=1e-5)
    assert tokenizer.vocab_size > 0
