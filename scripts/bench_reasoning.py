#!/usr/bin/env python3
"""Held-out reasoning benchmark: reformulated problems with verified answers.

    .venv/bin/python scripts/bench_reasoning.py --backend tinyllm_run \
        --model runs/sft_v9_jarvis2:best --tools
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cases import Case  # noqa: E402
from bench_neutral import Generator, score_case  # noqa: E402
from reasoning_gen import eval_problems  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["tinyllm", "tinyllm_run"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--count", type=int, default=63)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    gen = Generator(args.backend, args.model, args.max_tokens, args.temperature, 1)
    gen.tools = args.tools
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    results = []
    for i, p in enumerate(eval_problems(args.count)):
        case = Case(f"{p.family}_{i}", "reasoning", p.prompt, p.expect, p.forbid, max_words=40)
        answer = gen.generate([], p.prompt)
        passed, reasons = score_case(case, answer)
        by_family[p.family][1] += 1
        by_family[p.family][0] += int(passed)
        results.append({"id": case.id, "prompt": p.prompt, "answer": answer, "passed": passed,
                        "reasons": reasons})  # fmt: skip
    total = sum(r["passed"] for r in results)
    print(f"Model: {args.model} (tools={args.tools})")
    for family, (ok, n) in by_family.items():
        print(f"  {family:<10} {ok}/{n}")
    print(f"Reasoning score: {total}/{len(results)} ({total / len(results):.0%})")
    if args.json_out:
        args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
