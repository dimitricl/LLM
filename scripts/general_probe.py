#!/usr/bin/env python3
"""Measure small-talk leakage on general-purpose prompts.

The annealing passes teach conversational reflexes.  On a 46 M model they also
tend to overwrite general behaviour: "Donne-moi trois idées de repas simples"
starts answering "Tout va bien, merci."  This probe checks that specific
regression, which ``chat_regression.py`` deliberately does not cover.

    .venv/bin/python scripts/general_probe.py \
        --run-dir runs/sft_v8_conv --run-dir runs/sft_v8_anneal2

A prompt "leaks" when the answer is a conversational filler instead of an
attempt at the question.  Lower leak count is better; 0/10 is the target.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from tinyllm.chat import build_prompt
from tinyllm.sample import generate_ids, load_run_model

PROMPTS: list[str] = [
    "Explique en deux phrases ce qu'est la photosynthèse.",
    "Donne-moi trois idées de repas simples.",
    "Quelle est la capitale de la France ?",
    "Comment faire cuire du riz ?",
    "Écris un court poème sur la mer.",
    "Combien font 7 fois 8 ?",
    "Résume en une phrase : le chat dort sur le canapé depuis ce matin.",
    "Quels sont les avantages du vélo en ville ?",
    "Comment apprendre une langue étrangère ?",
    "À quoi sert une bibliothèque ?",
]

#: Openers that only make sense in small talk.  A general question answered
#: with one of these is a leak, even if the rest of the sentence is plausible.
LEAK_OPENERS = (
    "bonjour",
    "bonsoir",
    "salut",
    "hello",
    "tout va bien",
    "ca va",
    "ça va",
    "je vais bien",
    "de rien",
    "je t'en prie",
    "je vous en prie",
    "avec plaisir",
    "enchanté",
    "ravi de",
    "merci de demander",
)

LEAK_PHRASES = ("merci de demander", "et vous ?", "et toi ?")


def normalize(text: str) -> str:
    lowered = text.lower().replace("’", "'")
    plain = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return " ".join(plain.split())


def leaks(answer: str) -> str | None:
    """Return the offending pattern, or ``None`` if the answer looks on-topic."""
    text = normalize(answer)
    stripped = re.sub(r"^[^a-z0-9]+", "", text)

    for opener in LEAK_OPENERS:
        if stripped.startswith(normalize(opener)):
            return opener

    for phrase in LEAK_PHRASES:
        if normalize(phrase) in text:
            return phrase

    return None


def ask(model, tokenizer, cfg, prompt: str, args) -> tuple[str, int]:
    turns: list[dict] = []
    max_prompt_tokens = cfg.model.context_len - args.max_tokens
    prompt_ids = build_prompt(turns, prompt, tokenizer, max_prompt_tokens)
    remaining = min(args.max_tokens, cfg.model.context_len - len(prompt_ids))

    output_ids = generate_ids(
        model,
        prompt_ids,
        max_tokens=remaining,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        stop_at_eos=True,
        eos_id=tokenizer.eos_id,
        stop_on_user_token=True,
        user_id=tokenizer.user_id,
        seed=args.seed,
    )
    answer = tokenizer.decode(output_ids[len(prompt_ids):]).strip()
    return answer, len(output_ids) - len(prompt_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", action="append", required=True, help="repeatable"
    )
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=45)
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = {
        run: load_run_model(run, args.checkpoint) for run in args.run_dir
    }

    summary: dict[str, int] = {}

    for run in args.run_dir:
        model, tokenizer, cfg = models[run]
        print(f"\n=== {run} ({args.checkpoint})")
        leaked = 0

        for prompt in PROMPTS:
            answer, tokens = ask(model, tokenizer, cfg, prompt, args)
            pattern = leaks(answer)
            if pattern:
                leaked += 1
            mark = f"LEAK[{pattern}]" if pattern else "ok"
            print(f"[{mark:<18}] {prompt}")
            print(f"   -> {answer}")

        summary[run] = leaked
        print(f"\nFuite small talk : {leaked}/{len(PROMPTS)}")

    if len(summary) > 1:
        print("\n=== Résumé")
        for run, leaked in summary.items():
            print(f"{leaked}/{len(PROMPTS)}  {run}")


if __name__ == "__main__":
    main()
