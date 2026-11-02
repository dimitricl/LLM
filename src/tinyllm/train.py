import argparse
import json
import math
import shutil
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tinyllm.checkpoint import load_checkpoint, save_checkpoint
from tinyllm.config import RunConfig, TrainConfig, load_config, save_config
from tinyllm.dataset import TokenDataset
from tinyllm.model import TinyLM

PROMPTS = ["Once upon a time", "The little girl", "One day, the cat"]
_STOP = {"requested": False}


def _request_stop(signum, frame) -> None:
    _STOP["requested"] = True


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


def _device_info():
    return mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()


def _memory_function(name):
    return getattr(mx, name, None) or getattr(mx.metal, name)


def _peak_memory() -> int:
    return _memory_function("get_peak_memory")()


def _reset_peak_memory() -> None:
    _memory_function("reset_peak_memory")()


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


def evaluate_loss(
    model, path: str, context_len: int, batch_size: int, batches: int = 10
) -> float:
    dataset = TokenDataset(path, context_len, seed=0)
    if len(dataset.data) <= context_len + 1:
        return float("nan")
    losses = [float(loss_fn(model, dataset.next_batch(batch_size))) for _ in range(batches)]
    return sum(losses) / len(losses)


def rotate_checkpoints(ckpt_dir: str | Path, keep: int = 3) -> None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return
    step_dirs = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old in step_dirs[:-keep] if keep > 0 else step_dirs:
        shutil.rmtree(old)


def train(
    cfg: RunConfig,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    force: bool = False,
) -> dict:
    info = _device_info()
    memory_size = info["memory_size"]
    _memory_function("set_memory_limit")(int(0.7 * memory_size))
    _memory_function("set_cache_limit")(1 * 2**30)

    if max_steps is not None:
        cfg = replace(cfg, train=replace(cfg.train, max_steps=max_steps))

    run_dir = Path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.json")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    dtype = mx.bfloat16 if cfg.train.dtype == "bfloat16" else mx.float32
    model = TinyLM(cfg.model)
    work = TinyLM(cfg.model)
    optimizer = make_optimizer(cfg.train)
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    dataset = TokenDataset(cfg.data.train_bin, cfg.model.context_len, cfg.train.seed)

    step = 0
    best_val = float("inf")
    if resume is not None:
        meta = load_checkpoint(resume, model, optimizer)
        step = int(meta["step"])
        best_val = float(meta.get("best_val", best_val))
        dataset.load_state(meta["dataset"])

    log_path = run_dir / "log.jsonl"
    tokenizer = None
    tokenizer_path = Path(cfg.data.tokenizer_path)
    if tokenizer_path.is_file():
        from tinyllm.tokenizer import TokenizerWrapper

        tokenizer = TokenizerWrapper(tokenizer_path)
    tokens_per_step = (
        cfg.train.micro_batch * cfg.train.grad_accum * cfg.model.context_len
    )
    start = time.perf_counter()

    signal.signal(signal.SIGINT, _request_stop)
    _STOP["requested"] = False

    while step < cfg.train.max_steps:
        _reset_peak_memory()
        learning_rate = lr_at_step(step, cfg.train)
        optimizer.learning_rate = learning_rate
        accumulated = None
        work.update(cast_tree(model.parameters(), dtype))
        for _ in range(cfg.train.grad_accum):
            batch = dataset.next_batch(cfg.train.micro_batch)
            loss, grads = loss_and_grad(work, batch)
            grads = cast_tree(grads, mx.float32)
            accumulated = (
                grads
                if accumulated is None
                else tree_map(lambda a, b: a + b, accumulated, grads)
            )
            mx.eval(accumulated, loss)
            peak_mb = _peak_memory() / 2**20
            if peak_mb > 0.8 * memory_size / 2**20:
                raise SystemExit(
                    f"peak memory {peak_mb:.0f} MB > {0.8 * memory_size / 2**20:.0f} MB; "
                    "reduce micro_batch and increase grad_accum"
                )
        accumulated = tree_map(lambda g: g / cfg.train.grad_accum, accumulated)
        accumulated, norm = clip_grads(accumulated, cfg.train.grad_clip)
        optimizer.update(model, accumulated)
        mx.eval(model.parameters(), optimizer.state)
        step += 1
        elapsed = time.perf_counter() - start

        if step % cfg.train.log_interval == 0:
            record = {
                "step": step,
                "tokens": step * tokens_per_step,
                "loss": float(loss),
                "lr": learning_rate,
                "grad_norm": float(norm),
                "tokens_per_s": step * tokens_per_step / elapsed,
                "peak_memory_mb": _peak_memory() / 2**20,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        if tokenizer is not None and step % cfg.train.sample_interval == 0:
            from tinyllm.sample import generate

            samples_dir = run_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            blocks = []
            for prompt in PROMPTS:
                text = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=120,
                    temperature=0.8,
                    top_p=0.9,
                    seed=step,
                )
                blocks.append(f"--- {prompt}\n{text}\n")
            (samples_dir / f"step_{step:07d}.txt").write_text("\n".join(blocks))

        if step == 100 and not force and step < cfg.train.max_steps:
            estimated_hours = elapsed / step * cfg.train.max_steps / 3600
            if estimated_hours > cfg.train.max_hours:
                raise SystemExit(
                    f"estimated {estimated_hours:.1f}h > max_hours="
                    f"{cfg.train.max_hours}; use --force to override"
                )

        val_loss = None
        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
            val_loss = evaluate_loss(
                model,
                cfg.data.val_bin,
                cfg.model.context_len,
                cfg.train.micro_batch,
            )

        if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps:
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
            meta = {
                "step": step,
                "best_val": best_val,
                "dataset": dataset.state(),
                "elapsed_s": elapsed,
            }
            save_checkpoint(ckpt_dir / f"step_{step:07d}", model, optimizer, meta)
            save_checkpoint(ckpt_dir / "latest", model, optimizer, meta)
            if val_loss is not None and val_loss <= best_val:
                save_checkpoint(ckpt_dir / "best", model, optimizer, meta)
            rotate_checkpoints(ckpt_dir, keep=3)

        if _STOP["requested"]:
            print("interrupt received, saving checkpoint")
            break

    meta = {
        "step": step,
        "best_val": best_val,
        "dataset": dataset.state(),
        "elapsed_s": time.perf_counter() - start,
    }
    save_checkpoint(ckpt_dir / "latest", model, optimizer, meta)

    return {"step": step, "best_val": best_val}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = train(
        load_config(args.config),
        resume=args.resume,
        max_steps=args.max_steps,
        force=args.force,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
