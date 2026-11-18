#!/usr/bin/env python3
"""Run the neutral benchmark against a TinyLLM bundle or an MLX model.

    # TinyLLM quantized bundle (project venv)
    .venv/bin/python scripts/bench_neutral.py --backend tinyllm \
        --model exports/tinyllm-v9-lite/tinyllm-v9

    # TinyLLM run directory (full precision checkpoint): --model runs/x:best
    # LFM2.5 via mlx-lm, LFM3 via mlx-vlm (separate venv with mlx-vlm >= 0.7.11)
    python scripts/bench_neutral.py --backend mlx_lm  --model <dir>
    python scripts/bench_neutral.py --backend mlx_vlm --model <dir>

Scoring is regex based and deterministic; see ``bench_cases.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cases import CASES, Case  # noqa: E402


def normalize(text: str) -> str:
    """Lowercase, keep accents (regexes are accent aware), collapse spaces."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def score_case(case: Case, answer: str) -> tuple[bool, list[str]]:
    """Return ``(passed, reasons)`` for one answer."""
    text = normalize(answer)
    reasons: list[str] = []
    if not text:
        return False, ["empty answer"]
    if case.expect and not all(re.search(p, text) for p in case.expect):
        reasons.append("missing: " + " & ".join(case.expect))
    hit = [p for p in case.forbid if re.search(p, text)]
    if hit:
        reasons.append("forbidden: " + ", ".join(hit))
    words = len(answer.split())
    if words > case.max_words:
        reasons.append(f"too long ({words} > {case.max_words} words)")
    return not reasons, reasons


def summarize(results: list[dict]) -> dict[str, tuple[int, int]]:
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["passed"])
    return {k: (v[0], v[1]) for k, v in by_cat.items()}


class Generator:
    """Backend wrapper: ``generate(turns, prompt) -> str``."""

    def __init__(
        self,
        backend: str,
        model_path: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        adapter: str | None = None,
    ):
        self.backend, self.max_tokens, self.temperature, self.seed = (
            backend,
            max_tokens,
            temperature,
            seed,
        )
        if backend == "tinyllm":
            from tinyllm.quantize import load_model

            self.model, self.tok = load_model(model_path)
        elif backend == "tinyllm_run":
            from tinyllm.sample import load_run_model

            run_dir, _, checkpoint = model_path.partition(":")
            self.model, self.tok, _ = load_run_model(run_dir, checkpoint or "best")
        elif backend == "mlx_lm":
            from mlx_lm import load

            self.model, self.tok = load(model_path, adapter_path=adapter)
        elif backend == "mlx_vlm":
            from mlx_vlm import load

            self.model, self.proc = load(model_path, adapter_path=adapter)
            self.tok = getattr(self.proc, "tokenizer", self.proc)
        else:
            raise ValueError(f"unknown backend {backend}")

    tools = False

    def generate(self, turns: list[tuple[str, str]], prompt: str) -> str:
        if self.backend in ("tinyllm", "tinyllm_run"):
            from tinyllm.chat import build_prompt
            from tinyllm.sample import generate_ids

            history = [{"role": r, "content": c} for r, c in turns]
            ids = build_prompt(history, prompt, self.tok, 512 - self.max_tokens)
            if self.tools:
                from tinyllm.tool_runtime import generate_with_tools
                from tinyllm.tools import strip_tool_markup

                gen_ids, _ = generate_with_tools(
                    self.model,
                    self.tok,
                    ids,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=50,
                    repetition_penalty=1.05,
                    eos_id=self.tok.eos_id,
                    user_id=self.tok.user_id,
                    seed=self.seed,
                )
                return strip_tool_markup(self.tok.decode(gen_ids)).strip()
            out = generate_ids(
                self.model,
                ids,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=50,
                repetition_penalty=1.05,
                no_repeat_ngram_size=4,
                eos_id=self.tok.eos_id,
                stop_on_user_token=True,
                user_id=self.tok.user_id,
                seed=self.seed,
            )
            return self.tok.decode(out[len(ids) :]).strip()

        messages = [{"role": r, "content": c} for r, c in turns] + [
            {"role": "user", "content": prompt}
        ]
        try:
            text = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            text = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        if self.backend == "mlx_lm":
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            return generate(
                self.model,
                self.tok,
                prompt=text,
                max_tokens=self.max_tokens,
                sampler=make_sampler(temp=self.temperature, top_k=50),
            ).strip()
        from mlx_vlm import generate

        result = generate(
            self.model,
            self.proc,
            text,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=50,
            repetition_penalty=1.05,
            verbose=False,
        )
        return (result.text if hasattr(result, "text") else str(result)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", required=True, choices=["tinyllm", "tinyllm_run", "mlx_lm", "mlx_vlm"]
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tools", action="store_true", help="run <tool> calls (tinyllm backends)")
    parser.add_argument("--adapter", help="LoRA adapter dir (mlx_lm and mlx_vlm backends)")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true", help="print every answer")
    args = parser.parse_args()

    gen = Generator(
        args.backend, args.model, args.max_tokens, args.temperature, args.seed, args.adapter
    )
    gen.tools = args.tools
    results = []
    for case in CASES:
        start = time.perf_counter()
        answer = gen.generate(list(case.context), case.prompt)
        passed, reasons = score_case(case, answer)
        results.append(
            {
                "id": case.id,
                "category": case.category,
                "prompt": case.prompt,
                "answer": answer,
                "passed": passed,
                "reasons": reasons,
                "seconds": round(time.perf_counter() - start, 2),
            }
        )
        if args.verbose or not passed:
            mark = "OK  " if passed else "FAIL"
            print(f"[{mark}] {case.id:<18} {answer[:110]!r}")
            for reason in reasons:
                print(f"         -> {reason}")

    total = sum(r["passed"] for r in results)
    print(f"\nModel: {args.model} ({args.backend})")
    for cat, (ok, n) in summarize(results).items():
        print(f"  {cat:<14} {ok}/{n}")
    print(f"Score: {total}/{len(results)} ({total / len(results):.0%})")
    if args.json_out:
        args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
