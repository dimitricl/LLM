#!/usr/bin/env python3
"""Self-distilled replay: the base model answers general prompts itself.

Training a LoRA on the base model's own answers keeps its behaviour intact while
the persona rows teach identity.  Needs ``mlx-vlm`` (LFM3 loads only through it).

    python scripts/lfm/gen_replay.py --model <LFM3-1.7B-MLX-4bit dir> \
        --source data/sft/downloaded/raw/combined_with_teacher.jsonl --out replay.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def sample_prompts(source: Path, count: int, seed: int) -> list[str]:
    """Reservoir-sample short first-turn user prompts from a messages JSONL."""
    rng, pool, seen = random.Random(seed), [], 0
    for line in source.open():
        messages = json.loads(line)["messages"]
        if (
            not messages
            or messages[0]["role"] != "user"
            or not 8 < len(messages[0]["content"]) < 200
        ):
            continue
        seen += 1
        if len(pool) < count:
            pool.append(messages[0]["content"])
        elif (j := rng.randrange(seen)) < count:
            pool[j] = messages[0]["content"]
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=110)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from mlx_vlm import generate, load

    model, proc = load(args.model)
    tok = getattr(proc, "tokenizer", proc)
    kept = 0
    with args.out.open("w") as out:
        for prompt in sample_prompts(args.source, args.count, args.seed):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            result = generate(
                model,
                proc,
                text,
                max_tokens=args.max_tokens,
                temperature=0.3,
                top_k=50,
                repetition_penalty=1.05,
                verbose=False,
            )
            answer = (result.text if hasattr(result, "text") else str(result)).strip()
            if len(answer) > 5 and "</think>" not in answer:
                row = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept {kept} replay rows in {args.out}")


if __name__ == "__main__":
    main()
