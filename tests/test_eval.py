import math

import mlx.core as mx
import mlx.nn as nn

from tinyllm.data import write_bin
from tinyllm.eval import evaluate_bpb


class UniformModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, tokens: mx.array) -> mx.array:
        return mx.zeros((*tokens.shape, self.vocab_size))


class FakeTokenizer:
    def token_bytes(self, token_id: int) -> int:
        return 1


def test_uniform_model_bpb_equals_log2_vocab(tmp_path):
    vocab = 64
    path = tmp_path / "val.bin"
    write_bin([list(range(9))], path)
    result = evaluate_bpb(
        UniformModel(vocab), path, FakeTokenizer(), context_len=4, batch_size=2
    )
    assert abs(result["bpb"] - math.log2(vocab)) < 1e-4
    assert result["tokens"] == 8
    assert result["bytes"] == 8


def test_short_file_returns_nan(tmp_path):
    path = tmp_path / "empty.bin"
    write_bin([[1, 2, 3]], path)
    result = evaluate_bpb(UniformModel(8), path, FakeTokenizer(), context_len=8)
    assert math.isnan(result["bpb"])
