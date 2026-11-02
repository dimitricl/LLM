import json

import pytest

from tinyllm.config import ModelConfig, load_config


def test_head_dim_and_valid_config():
    cfg = ModelConfig(d_model=512, n_heads=8)
    assert cfg.head_dim == 64


def test_invalid_head_divisibility_raises():
    with pytest.raises(ValueError, match="d_model must be divisible"):
        ModelConfig(d_model=513, n_heads=8)


def test_invalid_kv_divisibility_raises():
    with pytest.raises(ValueError, match="n_heads must be divisible"):
        ModelConfig(n_heads=8, n_kv_heads=3)


def test_load_config_with_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_name": "smoke"}))
    cfg = load_config(path)
    assert cfg.run_name == "smoke"
    assert cfg.model.vocab_size == 16384
    assert cfg.train.micro_batch == 16
    assert cfg.data.tokenizer_path == "data/tokenizer.json"
