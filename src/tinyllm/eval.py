import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tinyllm.data import read_bin


def evaluate_bpb(
    model,
    path: str | Path,
    tokenizer,
    context_len: int,
    batch_size: int = 8,
    max_batches: int | None = None,
) -> dict:
    data = read_bin(path)
    if len(data) <= context_len + 1:
        return {"bpb": float("nan"), "tokens": 0, "bytes": 0}
    n_sequences = (len(data) - 1) // context_len

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    batches = 0
    for start in range(0, n_sequences, batch_size):
        indices = range(start, min(start + batch_size, n_sequences))
        rows = np.stack(
            [data[i * context_len : (i + 1) * context_len + 1] for i in indices]
        ).astype(np.int32)
        batch = mx.array(rows)
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        total_nll += float(nn.losses.cross_entropy(logits, targets, reduction="sum"))
        total_tokens += targets.size
        for row in np.array(targets):
            total_bytes += sum(tokenizer.token_bytes(int(token)) for token in row)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if total_bytes == 0:
        return {"bpb": float("nan"), "tokens": total_tokens, "bytes": 0}
    return {
        "bpb": total_nll / (math.log(2) * total_bytes),
        "tokens": total_tokens,
        "bytes": total_bytes,
    }


def main() -> None:
    from tinyllm.sample import load_run_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--val-bin")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()

    model, tokenizer, cfg = load_run_model(args.run_dir, args.checkpoint)
    result = evaluate_bpb(
        model,
        args.val_bin or cfg.data.val_bin,
        tokenizer,
        args.context_len or cfg.model.context_len,
        max_batches=args.batches,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
