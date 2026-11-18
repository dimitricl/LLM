
"""Supervised fine-tuning for TinyLLM. Keeps the pretraining run untouched."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import psutil
from mlx.utils import tree_flatten, tree_map

from tinyllm.checkpoint import load_checkpoint, save_checkpoint
from tinyllm.config import load_config
from tinyllm.dataset import TokenDataset
from tinyllm.model import TinyLM
from tinyllm.sft_data import SUPPORTED_SEQUENCE_LAYOUTS
from tinyllm.sft_dataset import SFTSequenceDataset

_STOP = {"requested": False}


def request_stop(signum, frame) -> None:
    _STOP["requested"] = True


def get_memory_usage() -> dict:
    """Get current memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "rss_gb": mem_info.rss / (1024**3),
        "vms_gb": mem_info.vms / (1024**3),
    }


def cast_tree(tree, dtype):
    return tree_map(lambda value: value.astype(dtype), tree)


def grad_norm(grads) -> mx.array:
    total = mx.zeros(())
    for _, gradient in tree_flatten(grads):
        total = total + mx.sum(gradient.astype(mx.float32) ** 2)
    return mx.sqrt(total)


def clip_grads(grads, max_norm: float):
    norm = grad_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda grad: grad * scale, grads), norm


def masked_loss(model, tokens: mx.array, loss_mask: mx.array) -> mx.array:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]

    losses = nn.losses.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    mask = loss_mask.astype(losses.dtype)
    token_count = mx.sum(mask)

    return mx.sum(losses * mask) / mx.maximum(token_count, 1.0)


class MixedBatcher:
    """Mix assistant-only SFT batches with optional all-token replay."""

    def __init__(self, sft, replay, replay_ratio: float, seed: int):
        if not 0.0 <= replay_ratio < 0.5:
            raise ValueError("replay_ratio must be in [0, 0.5).")

        self.sft = sft
        self.replay = replay
        self.replay_ratio = replay_ratio if replay is not None else 0.0
        self.rng = np.random.default_rng(seed)

    def next_batch(self, batch_size: int):
        token_rows = []
        mask_rows = []

        for _ in range(batch_size):
            use_replay = (
                self.replay is not None
                and self.rng.random() < self.replay_ratio
            )

            if use_replay:
                tokens = self.replay.next_batch(1)
                mask = mx.ones(
                    (1, tokens.shape[1] - 1),
                    dtype=mx.float32,
                )
            else:
                tokens, mask = self.sft.next_batch(1)

            token_rows.append(tokens)
            mask_rows.append(mask)

        return (
            mx.concatenate(token_rows, axis=0),
            mx.concatenate(mask_rows, axis=0),
        )

    def state(self) -> dict:
        return {
            "sft": self.sft.state(),
            "replay": None if self.replay is None else self.replay.state(),
            "rng": self.rng.bit_generator.state,
        }

    def load_state(self, state: dict) -> None:
        self.sft.load_state(state["sft"])

        if self.replay is not None and state.get("replay") is not None:
            self.replay.load_state(state["replay"])

        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]


def learning_rate(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_ratio: float = 0.03,
) -> float:
    warmup = max(1, int(total_steps * warmup_ratio))

    if step < warmup:
        return base_lr * (step + 1) / warmup

    progress = (step - warmup) / max(1, total_steps - warmup)

    return base_lr * (
        0.1
        + 0.9 * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )
    )


def packed_context_len(
    token_path: str | Path,
    requested_context_len: int | None,
    model_context_len: int,
) -> int:
    token_path = Path(token_path)
    metadata_path = token_path.parent / "meta.json"

    packed_len = None

    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if metadata.get("sequence_layout") not in SUPPORTED_SEQUENCE_LAYOUTS:
            raise ValueError("Unsupported SFT dataset layout.")

        packed_len = int(metadata["context_len"])

    context_len = requested_context_len or packed_len or model_context_len

    if packed_len is not None and context_len != packed_len:
        raise ValueError(
            f"Requested context_len={context_len}, "
            f"but packed dataset uses {packed_len}."
        )

    if not 1 <= context_len <= model_context_len:
        raise ValueError(
            f"context_len must be between 1 and {model_context_len}."
        )

    return context_len


def evaluate(
    model,
    dataset: SFTSequenceDataset,
    batch_size: int,
    batches: int,
) -> float:
    if batch_size < 1 or batches < 1:
        raise ValueError("Evaluation batch size and batches must be positive.")

    # Always evaluate from the beginning of the deterministic epoch order.
    original_state = dataset.state()
    eval_state = dict(original_state)
    eval_state["cursor"] = 0
    dataset.load_state(eval_state)

    total_loss = 0.0
    total_tokens = 0.0
    consumed = 0

    max_examples = min(len(dataset), batches * batch_size)

    try:
        while consumed < max_examples:
            current_batch = min(batch_size, max_examples - consumed)
            tokens, mask = dataset.next_batch(current_batch)

            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]

            per_token = nn.losses.cross_entropy(
                logits,
                targets,
                reduction="none",
            )

            weighted_loss = mx.sum(per_token * mask)
            token_count = mx.sum(mask)

            mx.eval(weighted_loss, token_count)

            total_loss += float(weighted_loss)
            total_tokens += float(token_count)
            consumed += current_batch

    finally:
        dataset.load_state(original_state)

    if total_tokens <= 0:
        raise ValueError("Evaluation dataset contains no supervised tokens.")

    return total_loss / total_tokens


def rotate_checkpoints(checkpoint_dir: Path, keep: int = 3) -> None:
    dirs = sorted(
        (
            path
            for path in checkpoint_dir.glob("step_*")
            if path.is_dir()
        ),
        key=lambda path: int(path.name.split("_")[1]),
    )

    for old in dirs[:-keep]:
        shutil.rmtree(old)


def train(args) -> dict:
    cfg = load_config(args.base_config)
    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float32

    context_len = packed_context_len(
        args.train_tokens,
        args.context_len,
        cfg.model.context_len,
    )

    model = TinyLM(cfg.model)

    # Load ONLY pretrained weights. SFT starts with a fresh optimizer.
    load_checkpoint(args.base_checkpoint, model, optimizer=None)

    work = TinyLM(cfg.model)
    work.update(cast_tree(model.parameters(), dtype))

    base_run_dir = Path(args.base_checkpoint).resolve().parent.parent
    run_dir = Path(args.output_dir).resolve()

    if run_dir == base_run_dir or base_run_dir in run_dir.parents:
        raise ValueError(
            "SFT output_dir must not be inside the pretrained run directory."
        )

    checkpoint_dir = run_dir / "checkpoints"

    if (
        checkpoint_dir.exists()
        and any(checkpoint_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            f"{checkpoint_dir} already contains checkpoints. "
            "Use --resume or choose another output directory."
        )

    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    sft = SFTSequenceDataset(
        args.train_tokens,
        args.train_mask,
        context_len,
        args.seed,
    )

    val_token_path = Path(args.val_tokens)
    val_mask_path = Path(args.val_mask)

    if not val_token_path.is_file() or not val_mask_path.is_file():
        raise FileNotFoundError("Validation token/mask files are missing.")

    if val_token_path.stat().st_size == 0 and val_mask_path.stat().st_size == 0:
        val_ds = None
        print("Validation disabled: empty validation split.", flush=True)
    elif val_token_path.stat().st_size == 0 or val_mask_path.stat().st_size == 0:
        raise ValueError("Validation token/mask files are inconsistent.")
    else:
        val_ds = SFTSequenceDataset(
            args.val_tokens,
            args.val_mask,
            context_len,
            seed=12345,
        )

    replay = None

    if args.replay_ratio > 0:
        if not args.replay_bin or not Path(args.replay_bin).is_file():
            raise FileNotFoundError(
                "Replay dataset is missing. Set --replay-ratio 0 to disable replay."
            )

        replay = TokenDataset(
            args.replay_bin,
            context_len,
            args.seed + 1,
        )

    batcher = MixedBatcher(
        sft,
        replay,
        args.replay_ratio,
        args.seed + 2,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_val = float("inf")

    if args.resume:
        meta = load_checkpoint(args.resume, model, optimizer)

        start_step = int(meta["step"])
        best_val = float(meta.get("best_val", best_val))

        if meta.get("batcher"):
            batcher.load_state(meta["batcher"])

        # Critical: synchronize the working model with resumed SFT weights.
        work.update(cast_tree(model.parameters(), dtype))
        mx.eval(work.parameters())

    run_config = vars(args).copy()
    run_config["resolved_context_len"] = context_len
    run_config["model_config"] = vars(cfg.model)

    (run_dir / "sft_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    log_path = run_dir / "log.jsonl"
    loss_and_grad = nn.value_and_grad(work, masked_loss)

    signal.signal(signal.SIGINT, request_stop)
    _STOP["requested"] = False

    start = time.perf_counter()
    tokens_per_step = args.micro_batch * args.grad_accum * context_len

    completed_step = start_step
    end_step = (
        min(args.steps, start_step + 1)
        if args.dry_run
        else args.steps
    )

    for step in range(start_step, end_step):
        current_lr = learning_rate(step, args.steps, args.lr)
        optimizer.learning_rate = current_lr

        # FP32 master weights are copied into the selected forward dtype.
        work.update(cast_tree(model.parameters(), dtype))

        accumulated = None
        total_loss = 0.0
        total_supervised_tokens = 0.0

        for _ in range(args.grad_accum):
            tokens, mask = batcher.next_batch(args.micro_batch)

            token_count = float(mx.sum(mask))

            loss, grads = loss_and_grad(work, tokens, mask)
            grads = cast_tree(grads, mx.float32)

            mx.eval(loss, grads)

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"Non-finite training loss: {loss_value}"
                )

            if token_count > 0:
                # Convert each micro-batch mean gradient into a gradient sum.
                weighted_grads = tree_map(
                    lambda grad: grad * token_count,
                    grads,
                )

                accumulated = (
                    weighted_grads
                    if accumulated is None
                    else tree_map(
                        lambda a, b: a + b,
                        accumulated,
                        weighted_grads,
                    )
                )

                total_loss += loss_value * token_count
                total_supervised_tokens += token_count

        if accumulated is None or total_supervised_tokens <= 0:
            raise ValueError(
                "No supervised assistant tokens in the accumulated step."
            )

        # Normalize once, by the actual number of supervised tokens.
        accumulated = tree_map(
            lambda grad: grad / total_supervised_tokens,
            accumulated,
        )

        accumulated, norm = clip_grads(accumulated, args.grad_clip)

        mx.eval(accumulated, norm)

        norm_value = float(norm)

        if not math.isfinite(norm_value):
            raise FloatingPointError(
                f"Non-finite gradient norm: {norm_value}"
            )

        optimizer.update(model, accumulated)
        mx.eval(model.parameters(), optimizer.state)

        completed_step = step + 1
        step_loss = total_loss / total_supervised_tokens

        if completed_step % args.log_interval == 0 or args.dry_run:
            elapsed = time.perf_counter() - start
            tokens_per_sec = (completed_step * tokens_per_step) / elapsed if elapsed > 0 else 0
            memory = get_memory_usage()

            record = {
                "step": completed_step,
                "tokens_seen": completed_step * tokens_per_step,
                "supervised_tokens_step": int(total_supervised_tokens),
                "loss": step_loss,
                "lr": current_lr,
                "grad_norm": norm_value,
                "elapsed_s": elapsed,
                "tokens_per_sec": tokens_per_sec,
                "memory_rss_gb": memory["rss_gb"],
                "memory_vms_gb": memory["vms_gb"],
            }

            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            print(json.dumps(record), flush=True)

        val_loss = None

        if (
            not args.dry_run
            and val_ds is not None
            and (
                completed_step % args.eval_interval == 0
                or completed_step == args.steps
            )
        ):
            val_loss = evaluate(
                model,
                val_ds,
                min(args.micro_batch, len(val_ds)),
                args.eval_batches,
            )

            elapsed = time.perf_counter() - start
            memory = get_memory_usage()

            eval_record = {
                "step": completed_step,
                "val_loss": val_loss,
                "elapsed_s": elapsed,
                "memory_rss_gb": memory["rss_gb"],
                "memory_vms_gb": memory["vms_gb"],
            }

            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_record) + "\n")

            print(json.dumps(eval_record), flush=True)

        should_save = not args.dry_run and (
            completed_step % args.save_interval == 0
            or completed_step % args.eval_interval == 0
            or completed_step == args.steps
            or _STOP["requested"]
        )

        if should_save:
            is_best = val_loss is not None and val_loss < best_val

            if is_best:
                best_val = val_loss

            meta = {
                "step": completed_step,
                "best_val": best_val,
                "batcher": batcher.state(),
                "elapsed_s": time.perf_counter() - start,
                "base_checkpoint": str(args.base_checkpoint),
                "context_len": context_len,
            }

            step_path = checkpoint_dir / f"step_{completed_step:07d}"

            save_checkpoint(step_path, model, optimizer, meta)
            save_checkpoint(checkpoint_dir / "latest", model, optimizer, meta)

            if is_best:
                save_checkpoint(checkpoint_dir / "best", model, optimizer, meta)

            rotate_checkpoints(checkpoint_dir, keep=3)

        if _STOP["requested"]:
            print(
                "SIGINT received: checkpoint saved; stopping.",
                flush=True,
            )
            break

    return {
        "step": completed_step,
        "best_val": best_val,
        "run_dir": str(run_dir),
        "context_len": context_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune TinyLLM on assistant conversations."
    )

    parser.add_argument(
        "--base-config",
        default="runs/phase1_tinystories/config.json",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="runs/phase1_tinystories/checkpoints/best",
    )

    parser.add_argument(
        "--train-tokens",
        default="data/sft/packed_v2/train.tokens.bin",
    )
    parser.add_argument(
        "--train-mask",
        default="data/sft/packed_v2/train.mask.bin",
    )
    parser.add_argument(
        "--val-tokens",
        default="data/sft/packed_v2/val.tokens.bin",
    )
    parser.add_argument(
        "--val-mask",
        default="data/sft/packed_v2/val.mask.bin",
    )

    parser.add_argument("--replay-bin", default="data/phase1/train.bin")
    parser.add_argument("--output-dir", default="runs/sft_v2")

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--context-len", type=int, default=None)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one optimizer step without evaluation or checkpoint saving.",
    )

    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--replay-ratio", type=float, default=0.05)

    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--resume",
        help="Resume from an SFT checkpoint, including optimizer state.",
    )

    args = parser.parse_args()

    if args.steps < 1 or args.micro_batch < 1 or args.grad_accum < 1:
        parser.error("steps, micro-batch and grad-accum must be positive.")

    if args.lr <= 0 or args.grad_clip <= 0:
        parser.error("Learning rate and gradient clipping must be positive.")

    if args.weight_decay < 0:
        parser.error("weight-decay cannot be negative.")

    if not 0 <= args.replay_ratio < 0.5:
        parser.error("replay-ratio must be in [0, 0.5).")

    if min(
        args.eval_interval,
        args.eval_batches,
        args.save_interval,
        args.log_interval,
    ) < 1:
        parser.error("Evaluation, save and log intervals must be positive.")

    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    main()

