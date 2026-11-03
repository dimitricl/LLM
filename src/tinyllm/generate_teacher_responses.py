"""Generate teacher responses for response-based distillation.

This script extracts user prompts from the SFT dataset, generates responses
using a large teacher model (e.g., Gemma-3:4B), and saves them in the same
format as the original SFT data for easy integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np


def extract_prompts_from_packed_data(
    tokens_path: Path,
    mask_path: Path,
    tokenizer,
    context_len: int,
    max_prompts: int = 10000,
) -> List[Dict]:
    """
    Extract user prompts from packed SFT data.

    Returns list of {"role": "user", "content": "...} prompts.
    """
    # Load the packed data
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")

    seq_len = context_len
    num_sequences = len(tokens) // seq_len

    print(f"Loading {num_sequences} sequences from {tokens_path}")

    prompts = []
    user_id = tokenizer.user_id
    assistant_id = tokenizer.assistant_id

    for seq_idx in range(min(num_sequences, max_prompts)):
        seq_tokens = tokens[seq_idx * seq_len : (seq_idx + 1) * seq_len]
        seq_mask = masks[seq_idx * seq_len : (seq_idx + 1) * seq_len]

        # Find USER token
        user_positions = np.where(seq_tokens == user_id)[0]

        if len(user_positions) == 0:
            continue

        # Get the first user prompt (simplified: take first USER token)
        user_pos = user_positions[0]

        # Extract tokens after USER until ASSISTANT or end
        content_tokens = []
        for i in range(user_pos + 1, seq_len):
            if seq_tokens[i] == assistant_id or seq_tokens[i] == 0:  # ASSISTANT or PAD
                break
            content_tokens.append(seq_tokens[i])

        if len(content_tokens) == 0:
            continue

        # Decode
        content = tokenizer.decode(content_tokens)

        if len(content.strip()) < 10:  # Skip very short prompts
            continue

        prompts.append({
            "role": "user",
            "content": content.strip()
        })

        if len(prompts) % 1000 == 0:
            print(f"Extracted {len(prompts)} prompts...")

    print(f"Total prompts extracted: {len(prompts)}")
    return prompts


def generate_teacher_responses(
    prompts: List[Dict],
    teacher_model,
    tokenizer,
    max_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 1,
) -> List[Dict]:
    """
    Generate responses using the teacher model.

    Returns list of {"role": "assistant", "content": "...} responses.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")

    responses = []

    for i, prompt in enumerate(prompts):
        user_content = prompt["content"]

        # Format prompt for the teacher model
        # Use simple Gemma chat format: "user\n{content}\nmodel\n"
        formatted_prompt = f"user\n{user_content}\nmodel\n"

        # Generate response
        try:
            from mlx_lm import generate

            response = generate(
                teacher_model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            responses.append({
                "role": "assistant",
                "content": response.strip()
            })

        except Exception as e:
            print(f"Error generating response for prompt {i}: {e}")
            responses.append({
                "role": "assistant",
                "content": ""  # Empty response on error
            })

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(prompts)} responses...")

    return responses


def save_as_sft_format(
    prompts: List[Dict],
    responses: List[Dict],
    output_path: Path,
) -> None:
    """
    Save prompts and responses in the same format as original SFT data.

    Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...}]}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, responses):
            if not response["content"]:  # Skip empty responses
                continue

            conversation = {
                "messages": [prompt, response]
            }
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Saved {len(prompts)} conversations to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate teacher responses for response-based distillation."
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
        "--tokenizer",
        default="data/tokenizer.json",
    )
    parser.add_argument(
        "--output",
        default="data/sft/downloaded/raw/teacher_responses.jsonl",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=10000,
        help="Maximum number of prompts to extract and process",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for teacher response generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )

    args = parser.parse_args()

    # Load TinyLM tokenizer for extracting prompts
    from tinyllm.tokenizer import TokenizerWrapper
    tinyllm_tokenizer = TokenizerWrapper(args.tokenizer)

    # Extract prompts from packed data
    print("Extracting prompts from packed SFT data...")
    prompts = extract_prompts_from_packed_data(
        Path(args.train_tokens),
        Path(args.train_mask),
        tinyllm_tokenizer,
        args.context_len,
        args.max_prompts,
    )

    if len(prompts) == 0:
        print("No prompts extracted. Exiting.")
        return

    # Load teacher model using mlx-lm
    try:
        from mlx_lm import load
    except ImportError:
        print("Error: mlx-lm not installed. Install with: pip install mlx-lm")
        return

    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model, teacher_tokenizer = load(args.teacher_model)

    # Generate responses
    responses = generate_teacher_responses(
        prompts,
        teacher_model,
        teacher_tokenizer,
        args.max_tokens,
        args.temperature,
    )

    # Save in SFT format
    save_as_sft_format(
        prompts,
        responses,
        Path(args.output),
    )

    print(f"\nDone! Generated {len(responses)} teacher responses.")
    print(f"Output saved to: {args.output}")
    print("\nNext steps:")
    print("1. Combine with existing SFT data:")
    print(f"   cat {args.output} >> data/sft/downloaded/raw/combined.jsonl")
    print("2. Re-pack the data:")
    print("   .venv/bin/python -m tinyllm.sft_data --input data/sft/downloaded/raw/combined.jsonl ...")
    print("3. Train the student model normally:")
    print("   .venv/bin/python -m tinyllm.sft_train ...")


if __name__ == "__main__":
    main()
