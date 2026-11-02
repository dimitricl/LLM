import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM
from tinyllm.train import (
    cast_tree,
    clip_grads,
    grad_norm,
    lr_at_step,
    loss_fn,
    make_optimizer,
)


def _cfg() -> tuple[ModelConfig, TrainConfig]:
    model = ModelConfig(
        vocab_size=320, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, context_len=16,
    )
    train = TrainConfig(lr=1e-2, max_steps=100, warmup_ratio=0.1, micro_batch=2)
    return model, train


def test_loss_is_scalar_and_near_ln_vocab_at_init():
    model_cfg, _ = _cfg()
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    tokens = mx.random.randint(0, model_cfg.vocab_size, (2, 9))
    loss = loss_fn(model, tokens)
    assert loss.shape == ()
    assert abs(float(loss) - math.log(model_cfg.vocab_size)) < 0.5


def test_grad_norm_of_known_tree():
    grads = {"a": mx.array([3.0, 4.0]), "b": mx.array([12.0])}
    assert mx.allclose(grad_norm(grads), mx.array(13.0), atol=1e-5)


def test_clip_grads_caps_norm():
    grads = {"w": mx.array([3.0, 4.0])}
    clipped, norm = clip_grads(grads, max_norm=1.0)
    assert mx.allclose(norm, mx.array(5.0), atol=1e-5)
    assert mx.allclose(grad_norm(clipped), mx.array(1.0), atol=1e-4)


def test_lr_schedule_endpoints():
    _, train_cfg = _cfg()
    warmup = int(train_cfg.warmup_ratio * train_cfg.max_steps)
    assert abs(lr_at_step(0, train_cfg) - train_cfg.lr / warmup) < 1e-9
    assert abs(lr_at_step(train_cfg.max_steps - 1, train_cfg) - train_cfg.lr * 0.1) < 1e-5


def test_one_step_reduces_loss_on_single_batch():
    model_cfg, train_cfg = _cfg()
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    work = TinyLM(model_cfg)
    optimizer = make_optimizer(train_cfg)
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    batch = mx.random.randint(0, model_cfg.vocab_size, (2, 9))

    initial = float(loss_fn(model, batch))
    for _ in range(20):
        work.update(cast_tree(model.parameters(), mx.bfloat16))
        loss, grads = loss_and_grad(work, batch)
        grads = cast_tree(grads, mx.float32)
        optimizer.update(model, grads)
    final = float(loss_fn(model, batch))
    assert final < initial * 0.9


def test_accumulated_grads_match_single_batch():
    model_cfg, _ = _cfg()
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    work = TinyLM(model_cfg)
    work.update(cast_tree(model.parameters(), mx.float32))
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    batch = mx.random.randint(0, model_cfg.vocab_size, (4, 9))

    _, full_grads = loss_and_grad(work, batch)
    _, grad_a = loss_and_grad(work, batch[:2])
    _, grad_b = loss_and_grad(work, batch[2:])
    averaged = tree_map(lambda a, b: (a + b) / 2, grad_a, grad_b)
    for (_, g_full), (_, g_avg) in zip(tree_flatten(full_grads), tree_flatten(averaged)):
        assert mx.allclose(g_full, g_avg, atol=1e-4)
