#!/usr/bin/env python3
"""Bench an Ollama model as a potential teacher for TinyLLM.

Three questions, three sections:

1. ``chat``    — does it answer the TinyLLM conversational battery correctly?
2. ``general`` — does it hold up on general questions (no small-talk leak)?
3. ``teacher`` — can it emit clean JSON training data, fast enough to matter?

    .venv/bin/python scripts/bench_teacher.py --model lfm3:1.7b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


regression = load_module("chat_regression", ROOT / "scripts" / "chat_regression.py")
probe = load_module("general_probe", ROOT / "scripts" / "general_probe.py")

TEACHER_SYSTEM = (
    "Tu génères des données d'entraînement pour un petit modèle de langage "
    "francophone. Tu réponds uniquement en JSON valide, sans texte autour."
)

TEACHER_PROMPT = """Génère 5 conversations courtes en français, au format JSON :
[{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
Sujets imposés : salutation, météo, remerciement, question sur tes capacités, au revoir.
Chaque réponse de l'assistant fait moins de 25 mots."""


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.3,
    num_predict: int = 300,
    timeout: int = 180,
) -> tuple[str, float, int]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # LFM3 is a reasoning model: without this, thinking eats the whole
        # ``num_predict`` budget and ``content`` comes back empty.
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": 1.1,
        },
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    elapsed = time.perf_counter() - start

    text = body.get("message", {}).get("content", "").strip()
    count = body.get("eval_count") or max(1, len(text.split()))
    return text, elapsed, count


def normalize(text: str) -> str:
    lowered = text.lower().replace("’", "'")
    plain = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return " ".join(plain.split())


def check(case, answer: str) -> list[str]:
    normalized = normalize(answer)
    reasons = []
    if not any(normalize(token) in normalized for token in case.expect_any):
        reasons.append("attendu: " + " / ".join(case.expect_any))
    forbidden = [t for t in case.forbid if normalize(t) in normalized]
    if forbidden:
        reasons.append("interdit: " + ", ".join(forbidden))
    if len(answer.split()) > case.max_words:
        reasons.append(f"trop long ({len(answer.split())} mots)")
    return reasons


def format_messages(case) -> list[dict]:
    messages = [{"role": role, "content": content} for role, content in case.context]
    messages.append({"role": "user", "content": case.prompt})
    return messages


def extract_json(text: str):
    """Parse a JSON array, tolerating the two errors LFM3 makes most often.

    In practice it emits one extra closing bracket, or omits the final brace.
    Both are cosmetic, so a couple of cheap candidates recover the data — but
    the caller still records whether strict parsing worked, because that rate
    is what decides whether a repair step is needed in the pipeline.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]

    attempts = [
        candidate,
        candidate[:-1],  # extra closing bracket
        candidate + "]}",  # missing final brace
        candidate.rstrip("]}") + "]}]",
    ]

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    print(f"Modèle : {args.model}\n")
    timings: list[float] = []
    report: dict = {"model": args.model, "chat": [], "general": [], "teacher": {}}

    print("=== 1. Conversation (16 cas)")
    passed = 0
    for case in regression.CASES:
        answer, elapsed, count = chat(args.base_url, args.model, format_messages(case))
        reasons = check(case, answer)
        timings.append(count / elapsed if elapsed else 0.0)
        passed += not reasons
        mark = "OK  " if not reasons else "FAIL"
        print(f"[{mark}] {case.name:<26} {answer[:100]}")
        for reason in reasons:
            print(f"         -> {reason}")
        report["chat"].append(
            {
                "name": case.name,
                "prompt": case.prompt,
                "answer": answer,
                "passed": not reasons,
                "reasons": reasons,
                "seconds": round(elapsed, 2),
            }
        )
    print(f"\nScore conversation : {passed}/{len(regression.CASES)}")

    print("\n=== 2. Questions générales (10 cas)")
    leaks = 0
    for prompt in probe.PROMPTS:
        answer, elapsed, count = chat(
            args.base_url, args.model, [{"role": "user", "content": prompt}]
        )
        timings.append(count / elapsed if elapsed else 0.0)
        pattern = probe.leaks(answer)
        leaks += bool(pattern)
        mark = f"LEAK[{pattern}]" if pattern else "ok"
        print(f"[{mark:<18}] {prompt}")
        print(f"   -> {answer[:160]}")
        report["general"].append(
            {
                "prompt": prompt,
                "answer": answer,
                "leak": pattern,
                "seconds": round(elapsed, 2),
            }
        )
    print(f"\nFuite small talk : {leaks}/{len(probe.PROMPTS)}")

    print("\n=== 3. Génération de données (JSON)")
    raw, elapsed, count = chat(
        args.base_url,
        args.model,
        [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": TEACHER_PROMPT},
        ],
        temperature=0.7,
        num_predict=1200,
    )
    timings.append(count / elapsed if elapsed else 0.0)
    data = extract_json(raw)
    strict = True
    try:
        json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
    except (json.JSONDecodeError, ValueError):
        strict = False
    valid = data is not None
    well_formed = 0
    turns = 0

    if valid:
        for item in data:
            messages = item.get("messages") if isinstance(item, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            roles = [m.get("role") for m in messages]
            contents = [str(m.get("content", "")) for m in messages]
            if roles[:2] == ["user", "assistant"] and all(contents):
                well_formed += 1
                turns += len(messages)

    print(f"JSON strict        : {strict}")
    print(f"JSON après répa.   : {valid}")
    print(f"Conversations OK   : {well_formed}/{len(data) if valid else 0}")
    print(f"Temps              : {elapsed:.1f} s")
    if valid:
        print(json.dumps(data[:2], ensure_ascii=False, indent=2))

    report["teacher"] = {
        "json_strict": strict,
        "json_valid": valid,
        "conversations_ok": well_formed,
        "turns": turns,
        "seconds": round(elapsed, 2),
        "raw": raw,
    }

    print("\n=== Résumé")
    print(f"conversation : {passed}/{len(regression.CASES)}")
    print(f"fuites       : {leaks}/{len(probe.PROMPTS)}")
    print(f"JSON         : {'ok' if valid else 'invalide'} ({well_formed} conversations)")
    if timings:
        print(f"débit        : {statistics.mean(timings):.0f} tok/s (moyenne)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"JSON         : {args.json_out}")


if __name__ == "__main__":
    main()
