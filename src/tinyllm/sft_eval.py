
"""Compare pretrained and SFT checkpoints on the same masked test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from tinyllm.checkpoint import load_checkpoint
from tinyllm.config import load_config
from tinyllm.model import TinyLM
from tinyllm.sft_data import SUPPORTED_SEQUENCE_LAYOUTS
from tinyllm.sft_dataset import SFTSequenceDataset


def resolve_context_len(
    test_tokens: str,
    requested_context_len: int | None,
    model_context_len: int,
) -> int:
    metadata_path = Path(test_tokens).parent / "meta.json"
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
            f"but test dataset uses {packed_len}."
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
        raise ValueError("batch-size and batches must be positive.")

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
        raise ValueError("Test dataset contains no supervised tokens.")

    return total_loss / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-config",
        default="runs/phase1_tinystories/config.json",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="runs/phase1_tinystories/checkpoints/best",
    )
    parser.add_argument(
        "--sft-checkpoint",
        default="runs/sft_v2/checkpoints/best",
    )
    parser.add_argument(
        "--test-tokens",
        default="data/sft/packed_v2/test.tokens.bin",
    )
    parser.add_argument(
        "--test-mask",
        default="data/sft/packed_v2/test.mask.bin",
    )
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=50)

    args = parser.parse_args()

    if args.batch_size < 1 or args.batches < 1:
        parser.error("batch-size and batches must be positive.")

    cfg = load_config(args.base_config)

    context_len = resolve_context_len(
        args.test_tokens,
        args.context_len,
        cfg.model.context_len,
    )

    results = {}

    # Each dataset uses the same seed and deterministic order.
    for name, checkpoint in (
        ("pretrained", args.base_checkpoint),
        ("sft", args.sft_checkpoint),
    ):
        model = TinyLM(cfg.model)
        load_checkpoint(checkpoint, model, optimizer=None)

        dataset = SFTSequenceDataset(
            args.test_tokens,
            args.test_mask,
            context_len,
            seed=2026,
        )

        results[name] = evaluate(
            model,
            dataset,
            min(args.batch_size, len(dataset)),
            args.batches,
        )

    results["delta"] = results["sft"] - results["pretrained"]
    results["context_len"] = context_len

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

