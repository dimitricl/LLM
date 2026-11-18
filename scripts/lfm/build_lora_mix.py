#!/usr/bin/env python3
"""Mix deduplicated persona rows with self-distilled replay into train/valid JSONL.

    python scripts/lfm/build_lora_mix.py --persona data/sft/persona_v1/persona.jsonl \
        --replay replay.jsonl --out-dir data/lora_mix
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_mix(persona: list[dict], replay: list[dict], persona_rows: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    unique = list({json.dumps(r, ensure_ascii=False, sort_keys=True): r for r in persona}.values())
    rng.shuffle(unique)
    rows = unique[:persona_rows] + replay
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--persona-rows", type=int, default=110)
    parser.add_argument("--valid-frac", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load = lambda p: [json.loads(line) for line in p.open()]  # noqa: E731
    rows = build_mix(load(args.persona), load(args.replay), args.persona_rows, args.seed)
    cut = int(len(rows) * (1 - args.valid_frac))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", rows[:cut]), ("valid", rows[cut:])):
        (args.out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        )
    print(f"train {cut} / valid {len(rows) - cut}")


if __name__ == "__main__":
    main()
