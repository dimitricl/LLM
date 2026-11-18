#!/usr/bin/env python3
"""Build a tool-use SFT dataset (messages JSONL) for the ``<tool>`` text format.

Assistant turns look like ``<tool>calc(17*3)</tool><result>51</result>17 x 3 font 51.``
Results are computed with ``tinyllm.tools`` so they are always correct.  About
25 % of rows need no tool, to avoid over-triggering.  At inference the runtime
stops at ``</tool>`` and injects the real ``<result>``.

    python scripts/build_tools_data.py --out data/sft/tools_v1/tools.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from tinyllm.tools import run_tool

MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
NO_TOOL = [
    ("Bonjour !", "Bonjour ! Comment puis-je t'aider ?"),
    ("Salut", "Salut ! Que puis-je faire pour toi ?"),
    ("Merci !", "Avec plaisir !"),
    (
        "Que sais-tu faire ?",
        "Je peux discuter, calculer avec un outil et t'aider à rédiger de courts textes.",
    ),
    (
        "Quel temps fait-il à Paris ?",
        "Je n'ai pas accès à la météo en temps réel, donc je préfère ne pas inventer.",
    ),
    ("Quelle est la capitale de la France ?", "La capitale de la France est Paris."),
]


def call(expr: str, today: date | None = None, name: str = "calc") -> str:
    arg = expr if name == "calc" else ""
    return f"<tool>{name}({arg})</tool><result>{run_tool(name, arg, today)}</result>"


def math_row(rng: random.Random) -> dict:
    kind = rng.choice(["mul", "add", "sub", "div", "pct", "double", "square", "mixed"])
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    if kind == "mul":
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        q = rng.choice([f"Combien font {a} x {b} ?", f"{a} fois {b} ?", f"Calcule {a} * {b}."])
        expr, text = f"{a}*{b}", f"{a} x {b} font"
    elif kind == "add":
        q = rng.choice([f"Combien font {a} + {b} ?", f"Calcule {a} + {b}.", f"{a} plus {b} ?"])
        expr, text = f"{a}+{b}", f"{a} + {b} font"
    elif kind == "sub":
        a, b = max(a, b), min(a, b)
        q = rng.choice([f"Combien font {a} - {b} ?", f"{a} moins {b} ?"])
        expr, text = f"{a}-{b}", f"{a} - {b} font"
    elif kind == "div":
        b = rng.randint(2, 12)
        a = b * rng.randint(2, 30)
        q = rng.choice([f"Combien font {a} divisé par {b} ?", f"{a} / {b} ?"])
        expr, text = f"{a}/{b}", f"{a} divisé par {b} font"
    elif kind == "pct":
        p, n = rng.choice([5, 10, 15, 20, 25, 50]), rng.randint(2, 40) * 10
        q = f"Combien font {p} % de {n} ?"
        expr, text = f"{n}*{p}/100", f"{p} % de {n} font"
    elif kind == "double":
        q = f"Quel est le double de {a} ?"
        expr, text = f"{a}*2", f"Le double de {a} est"
    elif kind == "square":
        a = rng.randint(2, 20)
        q = f"Combien font {a} au carré ?"
        expr, text = f"{a}**2", f"{a} au carré font"
    else:
        c = rng.randint(2, 9)
        q = f"Calcule ({a} + {b}) * {c}."
        expr, text = f"({a}+{b})*{c}", f"({a} + {b}) x {c} font"
    result = run_tool("calc", expr)
    answer = f"{call(expr)}{text} {result}."
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]}


def date_row(rng: random.Random) -> dict:
    today = date(2026, 1, 1) + timedelta(days=rng.randint(0, 800))
    q = rng.choice(
        ["Quelle date sommes-nous ?", "On est quel jour ?", "Quelle est la date d'aujourd'hui ?"]
    )
    pretty = f"{today.day} {MONTHS[today.month - 1]} {today.year}"
    answer = f"{call('', today, 'date')}Nous sommes le {pretty}."
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]}


def build(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        roll = rng.random()
        if roll < 0.65:
            rows.append(math_row(rng))
        elif roll < 0.75:
            rows.append(date_row(rng))
        else:
            q, a = rng.choice(NO_TOOL)
            rows.append(
                {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sft/tools_v1/tools.jsonl"))
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = build(args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} conversations to {args.out}")


if __name__ == "__main__":
    main()
