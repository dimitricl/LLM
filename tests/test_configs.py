"""Tests for configuration files."""
import json
from pathlib import Path

from tinyllm.config import load_config, ModelConfig, TrainConfig, DataConfig


def test_pretrain_v8_config_exists():
    """Verify pretrain_v8.json config file exists."""
    config_path = Path("configs/pretrain_v8.json")
    assert config_path.exists(), "configs/pretrain_v8.json not found"


def test_pretrain_v8_config_valid():
    """Verify pretrain_v8.json can be loaded and has correct structure."""
    cfg = load_config("configs/pretrain_v8.json")

    # Model config
    assert cfg.model.vocab_size == 24576
    assert cfg.model.d_model == 512
    assert cfg.model.n_layers == 12
    assert cfg.model.n_heads == 8
    assert cfg.model.n_kv_heads == 2
    assert cfg.model.ffn_hidden == 1408
    assert cfg.model.context_len == 512

    # Train config
    assert cfg.train.micro_batch == 4
    assert cfg.train.grad_accum == 64
    assert cfg.train.lr == 0.0006
    assert cfg.train.min_lr_ratio == 0.1
    assert cfg.train.warmup_ratio == 0.03
    assert cfg.train.weight_decay == 0.1
    assert cfg.train.grad_clip == 1.0
    assert cfg.train.max_steps == 4700
    assert cfg.train.eval_interval == 100
    assert cfg.train.checkpoint_interval == 100
    assert cfg.train.sample_interval == 500
    assert cfg.train.max_hours == 48.0
    assert cfg.train.dtype == "bfloat16"

    # Data config
    assert cfg.data.tokenizer_path == "data/v8/tokenizer.json"
    assert cfg.data.train_bin == "data/v8/pretrain/tokenized/train.bin"
    assert cfg.data.val_bin == "data/v8/pretrain/tokenized/val.bin"
    assert cfg.data.test_bin == "data/v8/pretrain/tokenized/test.bin"
    assert cfg.data.val_frac == 0.015
    assert cfg.data.test_frac == 0.015


def test_pretrain_v8_tokens_per_step():
    """Verify tokens per step calculation matches spec."""
    cfg = load_config("configs/pretrain_v8.json")
    tokens_per_step = (
        cfg.train.micro_batch * cfg.train.grad_accum * cfg.model.context_len
    )
    assert tokens_per_step == 4 * 64 * 512  # 131,072 tokens/step


def test_pretrain_v8_estimated_total_tokens():
    """Verify max_steps * tokens_per_step yields ~600M tokens."""
    cfg = load_config("configs/pretrain_v8.json")
    tokens_per_step = (
        cfg.train.micro_batch * cfg.train.grad_accum * cfg.model.context_len
    )
    total_tokens = cfg.train.max_steps * tokens_per_step
    # 4700 * 131072 = 616,038,400 ~ 613M tokens
    assert 610_000_000 <= total_tokens <= 620_000_000


def test_model_param_count():
    """Calculate and verify model parameter count."""
    cfg = ModelConfig(
        vocab_size=24576,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        ffn_hidden=1408,
        context_len=512,
    )

    # Embeddings: vocab_size * d_model
    embedding_params = cfg.vocab_size * cfg.d_model

    # Per layer:
    # - Attention: q_proj, k_proj, v_proj, o_proj
    # - MLP: gate_proj, up_proj, down_proj
    # - 2x RMSNorm
    q_proj = cfg.d_model * cfg.n_heads * cfg.head_dim
    k_proj = cfg.d_model * cfg.n_kv_heads * cfg.head_dim
    v_proj = cfg.d_model * cfg.n_kv_heads * cfg.head_dim
    o_proj = cfg.n_heads * cfg.head_dim * cfg.d_model
    attn_params = q_proj + k_proj + v_proj + o_proj

    gate_proj = cfg.d_model * cfg.ffn_hidden
    up_proj = cfg.d_model * cfg.ffn_hidden
    down_proj = cfg.ffn_hidden * cfg.d_model
    mlp_params = gate_proj + up_proj + down_proj

    norm_params = 2 * cfg.d_model  # 2 RMSNorm per layer

    layer_params = attn_params + mlp_params + norm_params
    total_layers_params = layer_params * cfg.n_layers

    # Final norm
    final_norm = cfg.d_model

    total_params = embedding_params + total_layers_params + final_norm

    # With vocab_size=24576, the model has ~46.4M parameters
    # (vs ~42M with vocab_size=16384)
    assert 46_000_000 <= total_params <= 47_000_000

    # Embeddings are ~12.5M (24576 * 512)
    assert 12_000_000 <= embedding_params <= 13_000_000
