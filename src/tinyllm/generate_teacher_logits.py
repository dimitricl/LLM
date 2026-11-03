"""Generate teacher logits for knowledge distillation.

This script loads a large teacher model (e.g., Gemma-3:4B) and generates
logits on the SFT dataset. These logits are saved and used during distillation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from tinyllm.sft_dataset import SFTSequenceDataset


def generate_teacher_logits(
    teacher_model,
    dataset: SFTSequenceDataset,
    output_dir: Path,
    batch_size: int = 4,
    dtype: str = "bfloat16",
) -> None:
    """
    Generate and save teacher logits for the entire dataset.

    Args:
        teacher_model: The teacher model (loaded via mlx-lm)
        dataset: SFT dataset
        output_dir: Directory to save logits
        batch_size: Batch size for inference
        dtype: Data type for logits
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sequences = len(dataset)
    vocab_size = teacher_model.config.vocab_size
    context_len = dataset.context_len

    print(f"Generating logits for {total_sequences} sequences...")
    print(f"Vocab size: {vocab_size}, Context length: {context_len}")
    print(f"Output dtype: {dtype}")

    # Pre-allocate arrays for all sequences
    # Shape: (total_sequences, context_len, vocab_size)
    # This might be too large for memory, so we'll process in chunks

    if dtype == "bfloat16":
        dtype_np = np.float16  # Store as float16 to save space
    else:
        dtype_np = np.float32

    processed = 0
    chunk_size = 1000  # Process 1000 sequences at a time

    for chunk_start in range(0, total_sequences, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_sequences)
        chunk_size_actual = chunk_end - chunk_start

        print(f"Processing sequences {chunk_start} to {chunk_end}...")

        # Reset dataset cursor
        original_state = dataset.state()
        chunk_state = dict(original_state)
        chunk_state["cursor"] = chunk_start
        dataset.load_state(chunk_state)

        # Generate logits for this chunk
        chunk_logits = []
        chunk_masks = []

        for i in range(chunk_size_actual):
            tokens, mask = dataset.next_batch(1)

            # Generate logits with teacher
            with mx.stream(mx.default_stream()):
                input_ids = mx.array(tokens[:, :-1], dtype=mx.int32)
                logits = teacher_model(input_ids)

                # Convert to numpy
                logits_np = logits.astype(dtype if dtype == "float32" else "bfloat16")
                logits_np = np.array(logits_np, dtype=dtype_np)

                chunk_logits.append(logits_np)
                chunk_masks.append(np.array(mask, dtype=np.uint8))

        # Save chunk
        chunk_logits = np.concatenate(chunk_logits, axis=0)
        chunk_masks = np.concatenate(chunk_masks, axis=0)

        chunk_path = output_dir / f"logits_chunk_{chunk_start:07d}_{chunk_end:07d}.npz"
        np.savez_compressed(
            chunk_path,
            logits=chunk_logits,
            masks=chunk_masks,
        )

        processed += chunk_size_actual
        print(f"Saved chunk {processed}/{total_sequences} sequences to {chunk_path}")

    # Save metadata
    metadata = {
        "total_sequences": total_sequences,
        "vocab_size": vocab_size,
        "context_len": context_len,
        "dtype": dtype,
        "chunk_size": chunk_size,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done! Logits saved to {output_dir}")
    print(f"Total size: {sum(f.stat().st_size for f in output_dir.glob('*.npz')) / (1024**3):.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate teacher logits for knowledge distillation."
    )

    parser.add_argument(
        "--teacher-model",
        required=True,
        help="Teacher model name (e.g., google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--train-tokens",
        default="data/sft/packed_v3/train.tokens.bin",
    )
    parser.add_argument(
        "--train-mask",
        default="data/sft/packed_v3/train.mask.bin",
    )
    parser.add_argument(
        "--val-tokens",
        default="data/sft/packed_v3/val.tokens.bin",
    )
    parser.add_argument(
        "--val-mask",
        default="data/sft/packed_v3/val.mask.bin",
    )
    parser.add_argument(
        "--output-dir",
        default="data/teacher_logits",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )

    args = parser.parse_args()

    # Load teacher model using mlx-lm
    try:
        from mlx_lm import load
    except ImportError:
        print("Error: mlx-lm not installed. Install with: pip install mlx-lm")
        return

    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model, tokenizer = load(args.teacher_model)

    # Load dataset
    dataset = SFTSequenceDataset(
        args.train_tokens,
        args.train_mask,
        args.context_len,
        seed=42,
    )

    # Generate logits
    output_dir = Path(args.output_dir)
    generate_teacher_logits(
        teacher_model,
        dataset,
        output_dir,
        args.batch_size,
        args.dtype,
    )


if __name__ == "__main__":
    main()
