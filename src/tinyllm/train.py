import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tinyllm.config import TrainConfig


def loss_fn(model: nn.Module, tokens: mx.array) -> mx.array:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def grad_norm(grads) -> mx.array:
    total = mx.zeros(())
    for _, gradient in tree_flatten(grads):
        total = total + mx.sum(gradient.astype(mx.float32) ** 2)
    return mx.sqrt(total)


def clip_grads(grads, max_norm: float):
    norm = grad_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm


def cast_tree(tree, dtype):
    return tree_map(lambda value: value.astype(dtype), tree)


def make_optimizer(cfg: TrainConfig) -> optim.AdamW:
    return optim.AdamW(
        learning_rate=cfg.lr,
        betas=[cfg.beta1, cfg.beta2],
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.warmup_ratio * cfg.max_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + (cfg.lr - min_lr) * cosine
