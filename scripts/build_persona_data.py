#!/usr/bin/env python3
"""Build a small persona SFT dataset (messages JSONL) from ``configs/persona.json``.

One canonical reply per intent, many user phrasings, plus multi-turn
consistency checks (the same question asked twice gets the same answer).

    python scripts/build_persona_data.py --out data/sft/persona_v1/persona.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IDENTITY_Q = [
    "Qui es-tu ?",
    "Tu es qui ?",
    "Comment tu t'appelles ?",
    "C'est quoi ton nom ?",
    "Présente-toi.",
    "Tu es quoi exactement ?",
    "Peux-tu te présenter ?",
    "Qui êtes-vous ?",
    "Quel est ton nom ?",
    "T'es un humain ?",
    "Tu es une IA ?",
    "Who are you?",
    "Es-tu un robot ?",
    "Dis-moi qui tu es.",
    "Tu t'appelles comment ?",
]
CAPABILITY_Q = [
    "Que sais-tu faire ?",
    "Tu sais faire quoi ?",
    "À quoi tu sers ?",
    "Qu'est-ce que tu peux faire ?",
    "Tu peux m'aider à quoi ?",
    "Quelles sont tes capacités ?",
    "Tu fais quoi ?",
    "What can you do?",
]
REALTIME_Q = [
    "Quel temps fait-il aujourd'hui ?",
    "Quelle heure est-il ?",
    "Quel est le cours du Bitcoin ?",
    "Qui a gagné le match hier ?",
    "Quelles sont les actualités du jour ?",
    "Il pleut à Paris ?",
    "Quel est le score du match en ce moment ?",
    "Quel jour sommes-nous ?",
]
CREATOR_Q = [
    "Qui t'a créé ?",
    "Qui t'a entraîné ?",
    "Tu viens d'où ?",
    "Qui est ton créateur ?",
    "Comment as-tu été fabriqué ?",
]
GREETINGS = ["Bonjour !", "Salut", "Coucou", "Bonsoir !", "Hello"]


def build(persona: dict, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    intents = [
        (IDENTITY_Q, persona["identity_reply"]),
        (CAPABILITY_Q, persona["capabilities_reply"]),
        (REALTIME_Q, persona["no_realtime_reply"]),
        (CREATOR_Q, persona["creator_reply"]),
    ]
    rows: list[dict] = []
    for i in range(count):
        questions, reply = intents[i % len(intents)]
        q = rng.choice(questions)
        messages = [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
        roll = rng.random()
        if roll < 0.25:  # greeting first, then the question
            messages = [
                {"role": "user", "content": rng.choice(GREETINGS)},
                {"role": "assistant", "content": "Bonjour ! Comment puis-je t'aider ?"},
            ] + messages
        elif roll < 0.40:  # same question again: identical answer
            q2 = rng.choice(questions)
            messages += [{"role": "user", "content": q2}, {"role": "assistant", "content": reply}]
        rows.append({"messages": messages})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", type=Path, default=Path("configs/persona.json"))
    parser.add_argument("--out", type=Path, default=Path("data/sft/persona_v1/persona.jsonl"))
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    persona = json.loads(args.persona.read_text())
    rows = build(persona, args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} conversations to {args.out}")


if __name__ == "__main__":
    main()
