# ruff: noqa: E501
"""Programmatic reasoning problems with verified answers ("thinking" data).

Every problem family has TRAIN templates and different EVAL templates so the
held-out benchmark measures reformulation, not memorized wording.  Answers and
tool results are computed by code.  Assistant format::

    <think>short steps, arithmetic through <tool>calc(..)</tool><result>..</result></think>Answer.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinyllm.tools import run_tool  # noqa: E402

PEOPLE = {
    "Léa": "elle", "Marie": "elle", "Sarah": "elle", "Emma": "elle", "Chloé": "elle",
    "Inès": "elle", "Paul": "il", "Tom": "il", "Hugo": "il", "Lucas": "il",
    "Nathan": "il", "Louis": "il",
}  # fmt: skip
OBJECTS = ["pommes", "oranges", "billes", "cartes", "livres", "bonbons", "stylos", "timbres"]
PETS = ["Rex", "Félix", "Nemo", "Milou", "Tigrou"]
# (singular, plural) with article-ready singular forms
INCLUSIONS = [
    (("un chat", "chats"), ("un animal", "animaux")),
    (("un chêne", "chênes"), ("un arbre", "arbres")),
    (("un carré", "carrés"), ("un rectangle", "rectangles")),
    (("un moineau", "moineaux"), ("un oiseau", "oiseaux")),
    (("un dauphin", "dauphins"), ("un mammifère", "mammifères")),
    (("un rosier", "rosiers"), ("un arbuste", "arbustes")),
]
DISJOINT = [
    (("un chat", "chats"), ("un oiseau", "oiseaux")),
    (("un carré", "carrés"), ("un cercle", "cercles")),
    (("un chêne", "chênes"), ("un rosier", "rosiers")),
    (("un dauphin", "dauphins"), ("un moineau", "moineaux")),
]
FAMILIES = ("ordering", "sequence", "time", "syllogism", "sub_add", "multiply", "share")
WEIGHTS = (0.22, 0.12, 0.12, 0.2, 0.12, 0.11, 0.11)


@dataclass(frozen=True)
class Problem:
    family: str
    prompt: str
    think: str
    answer: str
    expect: tuple[str, ...]
    forbid: tuple[str, ...] = ()

    @property
    def assistant(self) -> str:
        return f"<think>{self.think}</think>{self.answer}"


def tool(expr: str) -> str:
    return f"<tool>calc({expr})</tool><result>{run_tool('calc', expr)}</result>"


def _number(value: int | str) -> str:
    return rf"\b{value}\b"


def ordering(rng: random.Random, split: str) -> Problem:
    a, b, c = rng.sample(list(PEOPLE), 3)  # a > b > c
    templates = {
        "train": [
            (lambda x, y: f"{x} est plus grand que {y}", "", "Qui est le plus grand ?", "Qui est le plus petit ?"),
            (lambda x, y: f"{x} a plus de billes que {y}", "", "Qui en a le plus ?", "Qui en a le moins ?"),
            (lambda x, y: f"{y} est moins rapide que {x}", "", "Qui est le plus rapide ?", "Qui est le plus lent ?"),
        ],
        "eval": [
            (lambda x, y: f"{x} dépasse {y} en taille", "", "Qui est le plus grand ?", "Qui est le plus petit ?"),
            (lambda x, y: f"{x} arrive avant {y}", "Dans la course, ", "Qui arrive en premier ?", "Qui arrive en dernier ?"),
            (lambda x, y: f"{y} possède moins de cartes que {x}", "", "Qui en possède le plus ?", "Qui en possède le moins ?"),
        ],
    }[split]  # fmt: skip
    stmt, prefix, q_top, q_bottom = rng.choice(templates)
    first, second = stmt(a, b), stmt(b, c)
    if rng.random() < 0.5:
        first, second = second, first
    top = rng.random() < 0.5
    target, others = (a, (b, c)) if top else (c, (a, b))
    prompt = f"{prefix}{first} et {second}. {q_top if top else q_bottom}"
    think = f"{a} > {b} > {c}. Donc {target}."
    return Problem("ordering", prompt, think, f"{target}.", (_number(target.lower()),),
                   tuple(rf"\b{n.lower()}\b" for n in others))  # fmt: skip


def sequence(rng: random.Random, split: str) -> Problem:
    step = rng.randint(2, 9)
    increasing = rng.random() < 0.7
    start = rng.randint(1, 30) if increasing else rng.randint(40, 90)
    terms = [start + (step if increasing else -step) * i for i in range(4)]
    nxt = terms[-1] + (step if increasing else -step)
    seq = ", ".join(map(str, terms))
    prompt = rng.choice(
        {
            "train": [f"Quel est le nombre suivant : {seq} ?", f"Complète la suite : {seq}, ..."],
            "eval": [
                f"Continue la série {seq} : quel est le terme suivant ?",
                f"Voici une suite de nombres : {seq}. Quel nombre vient ensuite ?",
            ],
        }[split]
    )
    verb, sign = ("ajoute", "+") if increasing else ("retire", "-")
    think = f"On {verb} {step} à chaque fois. {tool(f'{terms[-1]}{sign}{step}')}"
    return Problem("sequence", prompt, think, f"{nxt}.", (_number(nxt),))


def _clock(hour: int, minute: int) -> str:
    return f"{hour}h{minute:02d}" if minute else f"{hour}h"


def time_problem(rng: random.Random, split: str) -> Problem:
    h, m = rng.randint(6, 19), rng.choice([0, 15, 30, 45])
    dh, dm = rng.randint(1, 3), rng.choice([0, 15, 30, 45])
    total = h * 60 + m + dh * 60 + dm
    end_h, end_m = divmod(total, 60)
    t1, dur = _clock(h, m), _clock(dh, dm)
    prompt = rng.choice(
        {
            "train": [f"Un train part à {t1} et roule {dur}. À quelle heure arrive-t-il ?"],
            "eval": [
                f"Un film commence à {t1} et dure {dur}. À quelle heure se termine-t-il ?",
                f"Je pars à {t1} pour un voyage de {dur}. À quelle heure vais-je arriver ?",
            ],
        }[split]
    )
    minutes = f"Minutes : {m} + {dm} = {m + dm}"
    if m + dm >= 60:
        hours = f"{minutes}, soit 1 h et {m + dm - 60} min. Heures : {h} + {dh} + 1 = {end_h}."
    else:
        hours = f"{minutes}. Heures : {h} + {dh} = {end_h}."
    result = _clock(end_h, end_m)
    think = f"{t1} + {dur}. {hours} Résultat : {result}."
    pattern = rf"\b{end_h}\s?h\s?{end_m:02d}\b" if end_m else rf"\b{end_h}\s?h(\s?00)?\b"
    return Problem("time", prompt, think, f"{result}.", (pattern,))


def syllogism(rng: random.Random, split: str) -> Problem:
    name = rng.choice(PETS)
    positive = rng.random() < 0.6
    if positive:
        (xs, xp), (ys, yp) = rng.choice(INCLUSIONS)
        prompt = rng.choice(
            {
                "train": [f"Tous les {xp} sont des {yp}. {name} est {xs}. Est-ce que {name} est {ys} ?"],
                "eval": [
                    f"Chaque {xs[3:]} est {ys}. {name} est {xs}. {name} est-il aussi {ys} ?",
                    f"Si tout {xs[3:]} est {ys}, et que {name} est {xs}, peut-on dire que {name} est {ys} ?",
                ],
            }[split]
        )  # fmt: skip
        think = f"Tous les {xp} sont des {yp}. {name} est {xs}, donc {name} est {ys}."
        return Problem(
            "syllogism", prompt, think, f"Oui, {name} est {ys}.", (r"\boui\b",), (r"\bnon\b",)
        )
    (xs, xp), (ys, yp) = rng.choice(DISJOINT)
    prompt = rng.choice(
        {
            "train": [f"Aucun {xs[3:]} n'est {ys}. {name} est {xs}. Est-ce que {name} est {ys} ?"],
            "eval": [
                f"Un {xs[3:]} n'est jamais {ys}. {name} est {xs}. {name} est-il {ys} ?",
                f"Les {xp} ne sont jamais des {yp}. {name} est {xs}. Est-ce {ys} ?",
            ],
        }[split]
    )  # fmt: skip
    think = f"Aucun {xs[3:]} n'est {ys}. {name} est {xs}, donc {name} n'est pas {ys}."
    return Problem(
        "syllogism", prompt, think, f"Non, {name} n'est pas {ys}.", (r"\bnon\b",), (r"\boui\b",)
    )


def sub_add(rng: random.Random, split: str) -> Problem:
    a = rng.randint(15, 60)
    b = rng.randint(3, a - 5)
    c = rng.randint(2, 20)
    obj, who = rng.choice(OBJECTS), rng.choice(list(PEOPLE))
    pron = PEOPLE[who]
    result = a - b + c
    prompt = rng.choice(
        {
            "train": [f"{who} a {a} {obj}. {pron.capitalize()} en perd {b}, puis en gagne {c}. Combien en a-t-{pron} maintenant ?"],
            "eval": [f"Dans une boîte, il y a {a} {obj}. On en retire {b}, puis on en ajoute {c}. Combien y en a-t-il à la fin ?"],
        }[split]
    )  # fmt: skip
    think = f"{a} - {b} + {c}. {tool(f'{a}-{b}+{c}')}"
    return Problem("sub_add", prompt, think, f"Il y en a {result}.", (_number(result),))


def multiply(rng: random.Random, split: str) -> Problem:
    k, m = rng.randint(2, 9), rng.randint(2, 12)
    obj = rng.choice(OBJECTS)
    prompt = rng.choice(
        {
            "train": [f"Il y a {k} sacs de {m} {obj} chacun. Combien de {obj} en tout ?"],
            "eval": [f"Chaque boîte contient {m} {obj}. Combien y a-t-il de {obj} dans {k} boîtes ?"],
        }[split]
    )  # fmt: skip
    think = f"{k} x {m}. {tool(f'{k}*{m}')}"
    return Problem("multiply", prompt, think, f"Il y a {k * m} {obj}.", (_number(k * m),))


def share(rng: random.Random, split: str) -> Problem:
    k, q = rng.randint(2, 8), rng.randint(2, 12)
    total = k * q
    obj, who = rng.choice(OBJECTS), rng.choice(list(PEOPLE))
    prompt = rng.choice(
        {
            "train": [f"{who} partage {total} {obj} entre {k} amis, à parts égales. Combien chacun en reçoit-il ?"],
            "eval": [f"On répartit {total} {obj} également dans {k} sacs. Combien y en a-t-il dans chaque sac ?"],
        }[split]
    )  # fmt: skip
    think = f"{total} / {k}. {tool(f'{total}/{k}')}"
    return Problem("share", prompt, think, f"Chacun en a {q}.", (_number(q),))


GENERATORS = {
    "ordering": ordering,
    "sequence": sequence,
    "time": time_problem,
    "syllogism": syllogism,
    "sub_add": sub_add,
    "multiply": multiply,
    "share": share,
}


def problems(count: int, seed: int, split: str, balanced: bool = False) -> list[Problem]:
    rng = random.Random(seed)
    if balanced:  # round-robin so every family is tested equally often
        families = [FAMILIES[i % len(FAMILIES)] for i in range(count)]
    else:
        families = rng.choices(FAMILIES, weights=WEIGHTS, k=count)
    return [GENERATORS[family](rng, split) for family in families]


def train_rows(count: int, seed: int = 11) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": p.prompt},
                {"role": "assistant", "content": p.assistant},
            ]
        }
        for p in problems(count, seed, "train")
    ]


def eval_problems(count: int = 63, seed: int = 2026) -> list[Problem]:
    return problems(count, seed, "eval", balanced=True)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Write reasoning SFT rows (messages JSONL).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = train_rows(args.count, args.seed)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} reasoning rows to {args.out}")


if __name__ == "__main__":
    main()
