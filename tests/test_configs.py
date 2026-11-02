from pathlib import Path

import pytest

from tinyllm.config import load_config

CONFIGS = [
    "configs/phase1_tinystories.json",
    "configs/phase2_mix.json",
    "configs/phase2_mix_extended.json",
]


@pytest.mark.parametrize("path", CONFIGS)
def test_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.model.vocab_size == 16384
    assert cfg.model.context_len == 512
    assert cfg.model.n_heads % cfg.model.n_kv_heads == 0
    assert cfg.train.max_steps > 0
    assert sum(cfg.data.sources.values()) == pytest.approx(1.0)
    assert Path(path).exists()


def test_phase2_defaults_are_larger_than_phase1():
    phase1 = load_config(CONFIGS[0])
    phase2 = load_config(CONFIGS[1])
    assert phase2.model.d_model > phase1.model.d_model
    assert phase2.model.n_layers > phase1.model.n_layers
