from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from tinyllm.checkpoint import load_checkpoint, save_checkpoint
from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM
from tinyllm.train import make_optimizer


def _setup():
    model_cfg = ModelConfig(
        vocab_size=320, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, context_len=16,
    )
    train_cfg = TrainConfig(lr=1e-2, max_steps=10, warmup_ratio=0.1)
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    optimizer = make_optimizer(train_cfg)
    return model, optimizer, train_cfg


def _train_step(model, work, optimizer, batch):
    loss_and_grad = nn.value_and_grad(work, lambda m, t: nn.losses.cross_entropy(
        m(t[:, :-1]), t[:, 1:], reduction="mean"
    ))
    work.update(tree_map(lambda value: value.astype(mx.float32), model.parameters()))
    _, grads = loss_and_grad(work, batch)
    optimizer.update(model, grads)


def test_save_then_load_restores_parameters_and_meta(tmp_path):
    model, optimizer, train_cfg = _setup()
    batch = mx.random.randint(0, 320, (2, 9))
    _train_step(model, TinyLM(model.cfg), optimizer, batch)
    before = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    save_checkpoint(tmp_path, model, optimizer, {"step": 1, "best_val": 2.5})
    model.update(tree_map(lambda value: value * 0.0, model.parameters()))
    meta = load_checkpoint(tmp_path, model, optimizer)

    after = dict(tree_flatten(model.parameters()))
    assert meta["step"] == 1
    assert all(mx.allclose(before[k], after[k]) for k in before)


def test_resume_reproduces_continuation(tmp_path):
    model_b, optimizer_b, _ = _setup()
    batch = mx.random.randint(0, 320, (2, 9))
    work_b = TinyLM(model_b.cfg)
    _train_step(model_b, work_b, optimizer_b, batch)
    before = {k: mx.array(v) for k, v in tree_flatten(model_b.parameters())}
    save_checkpoint(tmp_path, model_b, optimizer_b, {"step": 1})

    model_c, optimizer_c, _ = _setup()
    load_checkpoint(tmp_path, model_c, optimizer_c)

    after = dict(tree_flatten(model_c.parameters()))
    assert all(mx.allclose(before[k], after[k]) for k in before)


def test_load_missing_checkpoint_raises(tmp_path):
    model, optimizer, _ = _setup()
    try:
        load_checkpoint(tmp_path / "missing", model, optimizer)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
