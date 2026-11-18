#!/usr/bin/env python3
"""Reproducible conversational regression test for TinyLLM.

Runs a fixed battery of prompts against a checkpoint and prints a pass/fail
table, so runs can be compared without eyeballing chat output.

    .venv/bin/python scripts/chat_regression.py --run-dir runs/sft_v8_anneal2

Exit code is 1 when the score is below ``--min-score``.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from tinyllm.chat import build_prompt
from tinyllm.sample import generate_ids, load_run_model


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expect_any: tuple[str, ...]
    forbid: tuple[str, ...] = ("html5", "code html", "je suis désolé")
    context: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    max_words: int = 40


CASES: list[Case] = [
    Case("greeting_fr", "bonjour", ("bonjour", "salut")),
    Case("greeting_casual", "slt", ("bonjour", "salut")),
    Case("greeting_en", "hello", ("bonjour", "salut", "hello")),
    Case("greeting_evening", "bonsoir", ("bonsoir", "bonjour", "salut")),
    Case("how_are_you", "ça va ?", ("bien", "ça va")),
    Case("how_are_you_casual", "ca va ?", ("bien", "ça va")),
    Case("identity", "qui es-tu ?", ("tinyllm", "modèle", "assistant")),
    Case("identity_casual", "qui es tu ?", ("tinyllm", "modèle", "assistant")),
    Case(
        "capability",
        "que sais-tu faire ?",
        ("discuter", "peux", "questions", "aider"),
    ),
    Case(
        "capability_casual",
        "que sais tu faire ?",
        ("discuter", "peux", "questions", "aider"),
    ),
    Case("thanks", "merci", ("plaisir", "rien", "prière", "en prie", "souci")),
    Case("goodbye", "au revoir", ("revoir", "bientôt", "journée", "plus")),
    Case(
        "unknown_time",
        "quelle heure est-il ?",
        ("heure", "horloge", "sais pas", "accès", "connaissances", "information", "limité"),
        forbid=("html5", "il est 1", "calcul", "météo", "ciel", "capteurs"),
    ),
    Case(
        "multiturn_question_third",
        "que sais-tu faire ?",
        ("discuter", "peux", "questions", "aider"),
        context=(
            ("user", "bonjour"),
            ("assistant", "Bonjour !"),
            ("user", "ça va ?"),
            ("assistant", "Je vais bien, merci ! Et vous ?"),
        ),
    ),
    Case(
        "multiturn_identity_third",
        "qui es-tu ?",
        ("tinyllm", "modèle", "assistant"),
        context=(
            ("user", "hello"),
            ("assistant", "Bonjour !"),
            ("user", "ca va ?"),
            ("assistant", "Je vais bien, merci. Et vous ?"),
        ),
    ),
    Case(
        "multiturn_thanks_closes",
        "merci, à bientôt",
        ("plaisir", "rien", "revoir", "bientôt", "journée", "en prie"),
        context=(
            ("user", "bonjour"),
            ("assistant", "Bonjour !"),
            ("user", "que sais-tu faire ?"),
            ("assistant", "Je peux discuter et répondre à des questions simples."),
        ),
    ),
    # --- v3: free-form multi-turn, where the 2-3 turn dataset used to break.
    Case(
        "multiturn_followup_weather",
        "et demain ?",
        ("météo", "demain", "prévision", "sais pas", "accès", "information"),
        forbid=("html5", "code html"),
        context=(
            ("user", "salut"),
            ("assistant", "Bonjour ! Ravi de vous parler."),
            ("user", "quel temps fait-il ?"),
            (
                "assistant",
                "Je n'ai pas accès à la météo réelle, mais j'espère qu'il fait beau chez vous.",
            ),
        ),
    ),
    Case(
        "multiturn_turn5_capability",
        "que sais-tu faire ?",
        ("discuter", "peux", "questions", "aider"),
        context=(
            ("user", "bonjour"),
            ("assistant", "Bonjour ! Comment puis-je vous aider ?"),
            ("user", "ça va ?"),
            ("assistant", "Ça va bien, merci ! Et toi ?"),
            ("user", "qui es-tu ?"),
            ("assistant", "Je suis TinyLLM, un petit modèle de langage."),
        ),
    ),
    Case(
        "topic_switch_identity",
        "qui es-tu ?",
        ("tinyllm", "modèle", "assistant"),
        context=(
            ("user", "comment cuire du riz ?"),
            ("assistant", "Je ne sais pas cuisiner, désolé."),
            ("user", "et une recette simple ?"),
            ("assistant", "Je n'ai pas de recettes à te proposer."),
        ),
    ),
    # Les réponses "je ne sais pas" sont interchangeables entre thèmes tant
    # qu'on n'interdit pas le mauvais thème : sans ça, répondre "je ne sais pas
    # calculer" à "quelle heure est-il ?" passe le test par accident.
    Case(
        "unknown_math",
        "combien font 7 fois 8 ?",
        ("calcul", "sais pas", "doué", "limité", "connaissances", "information"),
        forbid=("html5", "capteurs", "ciel", "météo", "heure"),
    ),
    Case(
        "unknown_math_simple",
        "combien font 1+1 ?",
        ("calcul", "sais pas", "doué", "limité", "connaissances", "information"),
        forbid=("html5", "capteurs", "ciel", "météo", "heure"),
    ),
]


def normalize(text: str) -> str:
    lowered = text.lower().replace("’", "'")
    plain = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return " ".join(plain.split())


def run_case(model, tokenizer, cfg, case: Case, args) -> dict:
    turns = [
        {"role": role, "content": content} for role, content in case.context
    ]
    max_prompt_tokens = cfg.model.context_len - args.max_tokens
    prompt_ids = build_prompt(turns, case.prompt, tokenizer, max_prompt_tokens)
    remaining = min(
        args.max_tokens, cfg.model.context_len - len(prompt_ids)
    )

    start = time.perf_counter()
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
        seed=args.seed + len(case.context),
    )
    elapsed = time.perf_counter() - start

    answer = tokenizer.decode(output_ids[len(prompt_ids):]).strip()
    generated = len(output_ids) - len(prompt_ids)
    speed = generated / elapsed if elapsed > 0 else 0.0

    normalized = normalize(answer)
    matched = any(normalize(token) in normalized for token in case.expect_any)
    forbidden = [token for token in case.forbid if normalize(token) in normalized]
    words = len(answer.split())

    reasons = []
    if not matched:
        reasons.append("attendu: " + " / ".join(case.expect_any))
    if forbidden:
        reasons.append("interdit: " + ", ".join(forbidden))
    if words > case.max_words:
        reasons.append(f"trop long ({words} mots > {case.max_words})")

    return {
        "name": case.name,
        "prompt": case.prompt,
        "answer": answer,
        "passed": not reasons,
        "reasons": reasons,
        "words": words,
        "tokens": generated,
        "tok_per_s": round(speed, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, cfg = load_run_model(args.run_dir, args.checkpoint)

    print(f"Run        : {args.run_dir} ({args.checkpoint})")
    print(f"Paramètres : {model.num_parameters():,}")
    print(f"Décodage   : temp={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} rep={args.repetition_penalty}\n")

    results = []

    for case in CASES:
        result = run_case(model, tokenizer, cfg, case, args)
        results.append(result)
        mark = "OK  " if result["passed"] else "FAIL"
        print(f"[{mark}] {result['name']:<26} {result['answer']}")
        for reason in result["reasons"]:
            print(f"         -> {reason}")

    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    score = passed / total if total else 0.0

    print(f"\nScore : {passed}/{total} ({score:.0%})")
    speeds = [r["tok_per_s"] for r in results if r["tokens"] > 0]
    if speeds:
        print(f"Débit  : {sum(speeds) / len(speeds):.0f} tok/s (moyenne)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "run_dir": str(args.run_dir),
                    "checkpoint": args.checkpoint,
                    "score": score,
                    "passed": passed,
                    "total": total,
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"JSON   : {args.json_out}")

    if score < args.min_score:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
