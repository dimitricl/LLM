"""Neutral conversation benchmark cases (no TinyLLM-specific wording).

Each case: id, category, optional context turns, prompt, and a pass rule:
``expect`` (regexes, any must match the normalized answer), ``forbid``
(regexes, none may match) and ``max_words``.  Used to compare TinyLLM against
other models on identical prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REFUSAL = (
    r"pas acc[eè]s|n'ai pas acc[eè]s|ne peux pas|ne sais pas|temps r[eé]el|"
    r"n'ai pas (cette )?information|don't have|cannot|can't"
)


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    prompt: str
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    context: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    max_words: int = 80


def _c(id, category, prompt, expect=(), forbid=(), context=(), max_words=80):
    return Case(id, category, prompt, tuple(expect), tuple(forbid), tuple(context), max_words)


CASES: list[Case] = [
    # greetings and small talk
    _c("greet_fr", "conversation", "Bonjour !", [r"bonjour|salut"], max_words=30),
    _c("greet_casual", "conversation", "salut", [r"bonjour|salut|hello"], max_words=30),
    _c("greet_en", "conversation", "Hello!", [r"hello|hi\b|bonjour|salut"], max_words=30),
    _c("how_are_you", "conversation", "Comment ça va ?", [r"bien|ça va|merci"], max_words=40),
    _c(
        "thanks",
        "conversation",
        "Merci beaucoup !",
        [r"de rien|pas de quoi|plaisir|avec plaisir"],
        max_words=30,
    ),
    _c(
        "goodbye",
        "conversation",
        "Au revoir !",
        [r"revoir|bientôt|bonne|à plus|salut"],
        max_words=30,
    ),
    # arithmetic
    _c("math_mul", "calcul", "Combien font 17 x 3 ?", [r"\b51\b"]),
    _c("math_add", "calcul", "Calcule 12 + 30.", [r"\b42\b"]),
    _c("math_sub", "calcul", "Combien font 100 - 37 ?", [r"\b63\b"]),
    _c("math_mul2", "calcul", "9 fois 8 ?", [r"\b72\b"]),
    _c("math_div", "calcul", "Combien font 144 divisé par 12 ?", [r"\b12\b"]),
    _c("math_one", "calcul", "1+1 ?", [r"\b2\b|deux"]),
    _c("math_pct", "calcul", "Combien font 25 % de 80 ?", [r"\b20\b"]),
    _c("math_double", "calcul", "Quel est le double de 35 ?", [r"\b70\b"]),
    # facts
    _c("fact_australia", "faits", "Quelle est la capitale de l'Australie ?", [r"canberra"]),
    _c("fact_france", "faits", "Quelle est la capitale de la France ?", [r"paris"]),
    _c("fact_water", "faits", "Quelle est la formule chimique de l'eau ?", [r"h2o|h₂o"]),
    _c("fact_mars", "faits", "Quelle planète est surnommée la planète rouge ?", [r"mars"]),
    _c("fact_hugo", "faits", "Qui a écrit Les Misérables ?", [r"hugo"]),
    _c("fact_ocean", "faits", "Quel est le plus grand océan du monde ?", [r"pacifique|pacific"]),
    _c("fact_brazil", "faits", "Quelle langue parle-t-on au Brésil ?", [r"portugais"]),
    _c("fact_week", "faits", "Combien de jours y a-t-il dans une semaine ?", [r"\b7\b|sept"]),
    _c(
        "fact_photo",
        "faits",
        "Explique la photosynthèse en deux phrases.",
        [r"lumi[eè]re|soleil|solaire", r"oxyg[eè]ne|glucose|[eé]nergie"],
        max_words=70,
    ),
    # logic and reasoning
    _c(
        "logic_syllogism",
        "raisonnement",
        "Si tous les chats sont des animaux et que Félix est un chat, Félix est-il un animal ?",
        [r"\boui\b"],
    ),
    _c(
        "logic_train",
        "raisonnement",
        "Un train part à 14h et roule 2h30. À quelle heure arrive-t-il ?",
        [r"16\s?h\s?30|16:30|16h30"],
    ),
    _c(
        "logic_order",
        "raisonnement",
        "Paul est plus grand que Marie et Marie est plus grande que Léa. Qui est le plus petit ?",
        [r"l[eé]a"],
    ),
    _c("logic_seq", "raisonnement", "Quel est le nombre suivant : 2, 4, 6, 8 ?", [r"\b10\b|dix"]),
    # memory within the conversation
    _c(
        "mem_name",
        "mémoire",
        "Comment je m'appelle ?",
        [r"l[eé]a"],
        context=(("user", "Je m'appelle Léa."), ("assistant", "Enchanté Léa !")),
    ),
    _c(
        "mem_city",
        "mémoire",
        "Où est-ce que j'habite ?",
        [r"lyon"],
        context=(
            ("user", "J'habite à Lyon."),
            ("assistant", "Très bien, Lyon est une belle ville."),
        ),
    ),
    _c(
        "mem_color",
        "mémoire",
        "Quelle est ma couleur préférée ?",
        [r"vert"],
        context=(("user", "Ma couleur préférée est le vert."), ("assistant", "Noté, le vert.")),
    ),
    # no access to live information: must refuse, not invent
    _c(
        "live_weather",
        "ignorance",
        "Quel temps fait-il à Paris en ce moment ?",
        [REFUSAL],
        [r"\d\s?°"],
    ),
    _c(
        "live_time",
        "ignorance",
        "Quelle heure est-il maintenant ?",
        [REFUSAL],
        [r"\b\d{1,2}\s?[h:]\s?\d{2}\b"],
    ),
    _c(
        "live_bitcoin",
        "ignorance",
        "Quel est le cours du Bitcoin aujourd'hui ?",
        [REFUSAL],
        [r"\d{4,}\s?(\$|€|dollars|euros)"],
    ),
    _c("live_match", "ignorance", "Qui a gagné le match hier soir ?", [REFUSAL]),
    # translation
    _c(
        "tr_cat",
        "traduction",
        "Traduis en anglais : « Le chat dort sur le canapé. »",
        [r"cat", r"sleep"],
    ),
    _c("tr_hello", "traduction", "Comment dit-on « hello » en français ?", [r"bonjour|salut"]),
    _c("tr_thanks", "traduction", "Traduis en anglais : « Merci beaucoup ».", [r"thank"]),
    # instruction following
    _c(
        "inst_yesno",
        "instructions",
        "Réponds uniquement par oui ou non : le ciel est-il bleu ?",
        [r"\boui\b"],
        max_words=6,
    ),
    _c(
        "inst_word",
        "instructions",
        "Écris une phrase avec le mot soleil.",
        [r"soleil"],
        max_words=40,
    ),
    _c(
        "inst_short",
        "instructions",
        "En une phrase : pourquoi la mer est-elle salée ?",
        [r"sel|min[eé]raux"],
        max_words=45,
    ),
    _c(
        "inst_list",
        "instructions",
        "Donne-moi trois conseils pour bien dormir.",
        [r"1|premi|\*|-"],
        max_words=120,
    ),
    _c(
        "inst_poem",
        "instructions",
        "Écris un poème de deux vers sur la mer.",
        [r"mer|vague|oc[eé]an|flot"],
        max_words=40,
    ),
    # English
    _c("en_capital", "anglais", "What is the capital of Japan?", [r"tokyo"], max_words=40),
    _c("en_math", "anglais", "What is 2+2?", [r"\b4\b|four"], max_words=30),
    _c("en_hello", "anglais", "Say hello.", [r"hello|hi\b"], max_words=20),
]
