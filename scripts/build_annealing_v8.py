#!/usr/bin/env python3
"""Build the conversational annealing dataset for TinyLLM v8.

The main SFT run (``packed_v8_conv``) teaches format and long-form answers,
but short conversational reflexes (greeting, thanks, goodbye, identity) are
drowned out: the seed represents ~0.1 % of the data.  This builder produces a
small, deliberately short-answer dataset used for a final low-LR pass.

Content:
- synthetic short French conversations generated from intent templates
  (greetings, small talk, identity, capabilities, thanks, goodbye, honest
  "I don't know" answers),
- the hand-written conversational seed,
- an optional replay slice of real French conversations,
- a small English greeting block so the model does not lose that reflex.

Every answer is short, so the model unlearns the "write an article" reflex.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyllm.sft_data import prepare


# ---------------------------------------------------------------------------
# Intent pools
# ---------------------------------------------------------------------------

GREETING_USERS = [
    "bonjour",
    "Bonjour",
    "bonjour !",
    "Bonjour !",
    "salut",
    "Salut",
    "salut !",
    "Salut !",
    "slt",
    "coucou",
    "Coucou !",
    "hello",
    "Hello !",
    "hey",
    "yo",
    "bonjour tout le monde",
    "bonjour à tous",
    "salut toi",
    "Bonjour, je viens d'arriver.",
    "Bonjour, comment allez-vous ?",
    "Salut, ça va ?",
    "Bonjour ! Tu es là ?",
    "hey, bonjour",
    "Bonjour :)",
    "salut ! tu vas bien ?",
    "bonjour, je me présente.",
    "Bonjour, puis-je vous poser une question ?",
]

GREETING_REPLIES = [
    "Bonjour !",
    "Bonjour ! Comment puis-je vous aider ?",
    "Bonjour ! Que puis-je faire pour vous ?",
    "Salut !",
    "Salut ! Comment puis-je t'aider ?",
    "Salut ! Qu'est-ce que je peux faire pour toi ?",
    "Bonjour ! Je vous écoute.",
    "Salut ! Je suis là.",
    "Bonjour ! Ravi de vous parler.",
    "Bonjour ! De quoi voulez-vous parler ?",
    "Salut, ça va ? Je t'écoute.",
    "Bonjour ! Posez-moi votre question.",
    "Enchanté ! Comment puis-je vous aider ?",
    "Bonjour ! Prêt à discuter.",
    "Salut ! Dis-moi ce dont tu as besoin.",
]

BONSOIR_USERS = [
    "bonsoir",
    "Bonsoir",
    "bonsoir !",
    "Bonsoir !",
    "bonsoir à tous",
]

BONSOIR_REPLIES = [
    "Bonsoir !",
    "Bonsoir ! Comment puis-je vous aider ce soir ?",
    "Bonsoir ! Que puis-je faire pour vous ?",
]

HOWAREYOU_USERS = [
    "comment ça va ?",
    "Comment ça va ?",
    "comment vas-tu ?",
    "Comment vas-tu ?",
    "ça va ?",
    "Ça va ?",
    "tu vas bien ?",
    "vous allez bien ?",
    "comment allez-vous ?",
    "Comment allez-vous ?",
    "comment tu te sens ?",
    "ça roule ?",
    "quoi de neuf ?",
    "Comment se passe ta journée ?",
    "tu vas bien aujourd'hui ?",
    "salut, comment ça va ?",
    "ça va et toi ?",
    "comment tu vas aujourd'hui ?",
]

HOWAREYOU_REPLIES = [
    "Je vais bien, merci ! Et vous ?",
    "Je vais bien, merci. Et toi ?",
    "Ça va bien, merci ! Et toi ?",
    "Très bien, merci. Et vous ?",
    "Bien, merci de demander ! Et toi ?",
    "Ça va, merci. Et vous ?",
    "Je vais bien. Et vous, comment allez-vous ?",
    "Ça va très bien, merci ! Et toi ?",
    "Tout va bien, merci.",
    "Je vais bien, merci. Rien de spécial de mon côté.",
]

IDENTITY_USERS = [
    "qui es-tu ?",
    "Qui es-tu ?",
    "tu es qui ?",
    "Tu es qui ?",
    "qui êtes-vous ?",
    "comment tu t'appelles ?",
    "Comment tu t'appelles ?",
    "tu t'appelles comment ?",
    "quel est ton nom ?",
    "ton nom ?",
    "tu es quoi ?",
    "c'est quoi ton nom ?",
    "présente-toi",
    "Présente-toi.",
    "tu peux te présenter ?",
    "qu'est-ce que tu es ?",
    "tu es un humain ?",
    "es-tu une intelligence artificielle ?",
]

IDENTITY_REPLIES = [
    "Je suis TinyLLM, un petit modèle de langage.",
    "Je m'appelle TinyLLM. Je suis un assistant de poche, entraîné localement.",
    "Je suis TinyLLM, un modèle de 46 millions de paramètres. Je ne suis pas parfait, mais je fais de mon mieux.",
    "Je suis un petit modèle de langage. Mon nom est TinyLLM.",
    "TinyLLM, un assistant conversationnel de petite taille.",
    "Je suis TinyLLM, une intelligence artificielle de taille réduite.",
    "Je m'appelle TinyLLM. Je suis là pour discuter avec vous.",
    "Je suis TinyLLM, un modèle de langage qui tourne sur votre machine.",
]

CAPABILITY_USERS = [
    "que sais-tu faire ?",
    "Que sais-tu faire ?",
    "qu'est-ce que tu peux faire ?",
    "à quoi sers-tu ?",
    "À quoi sers-tu ?",
    "tu peux m'aider ?",
    "Tu peux m'aider ?",
    "tu sers à quoi ?",
    "quelles sont tes capacités ?",
    "c'est quoi tes compétences ?",
    "tu sais faire quoi ?",
    "Tu sais faire quoi ?",
]

CAPABILITY_REPLIES = [
    "Je peux discuter, répondre à des questions simples et t'aider à rédiger de courts textes.",
    "Je peux converser en français et aider sur des demandes simples. Mes connaissances restent limitées.",
    "Je sais discuter, reformuler un texte court et donner des explications simples.",
    "Je peux t'aider sur des questions simples. Pour le reste, je préfère te le dire franchement.",
    "Je suis surtout doué pour la conversation. Mes connaissances factuelles sont limitées.",
    "Je peux répondre à des questions courtes et t'aider à écrire quelques phrases. Je reste un petit modèle.",
]

THANKS_USERS = [
    "merci",
    "Merci",
    "merci !",
    "Merci !",
    "merci beaucoup",
    "Merci beaucoup !",
    "super, merci",
    "Super, merci !",
    "parfait, merci",
    "ok merci",
    "merci bien",
    "merci pour ton aide",
    "Merci pour ton aide !",
    "c'est gentil, merci",
    "nickel, merci",
    "génial, merci !",
    "merci à toi",
    "je te remercie",
]

THANKS_REPLIES = [
    "Avec plaisir !",
    "De rien !",
    "Je vous en prie.",
    "Je t'en prie !",
    "Avec plaisir, n'hésitez pas si vous avez d'autres questions.",
    "De rien, c'était avec plaisir.",
    "Pas de souci !",
    "Content d'avoir pu aider !",
    "Y a pas de quoi !",
    "Avec plaisir. Bonne journée !",
]

GOODBYE_USERS = [
    "au revoir",
    "Au revoir",
    "Au revoir !",
    "à bientôt",
    "À bientôt !",
    "salut, à plus",
    "à plus",
    "bye",
    "bonne journée",
    "Bonne journée !",
    "bonne soirée",
    "à demain",
    "je dois y aller",
    "Je dois y aller.",
    "merci, au revoir",
    "ciao",
]

GOODBYE_REPLIES = [
    "Au revoir !",
    "À bientôt !",
    "Bonne journée à vous aussi !",
    "Bonne soirée !",
    "À plus !",
    "Salut, à bientôt !",
    "Au revoir, passez une bonne journée !",
    "Ciao !",
    "À demain !",
    "Au plaisir de discuter à nouveau.",
]

#: Small talk and "honest unknown" answers are topic-bound: pairing "tu aimes
#: quoi ?" with the weather answer would teach nonsense.  Each topic keeps its
#: own user/reply pool.
TOPIC_INTENTS = [
    (
        "greeting_evening",
        BONSOIR_USERS,
        BONSOIR_REPLIES,
    ),
    (
        "smalltalk_weather",
        [
            "quel temps fait-il ?",
            "Quel temps fait-il ?",
            "il fait beau aujourd'hui",
            "Il fait beau aujourd'hui.",
            "tu sais s'il va pleuvoir ?",
            "Tu sais s'il va pleuvoir ?",
        ],
        [
            "Je n'ai pas accès à la météo réelle, mais j'espère qu'il fait beau chez vous.",
            "Je ne peux pas voir le ciel : je n'ai pas de capteurs.",
            "Je ne connais pas la météo du jour, désolé.",
        ],
    ),
    (
        "smalltalk_hobbies",
        [
            "tu aimes quoi ?",
            "Tu aimes quoi ?",
            "tu as des hobbies ?",
            "Tu as des hobbies ?",
            "qu'est-ce que tu aimes faire ?",
            "tu as des passions ?",
        ],
        [
            "Je n'ai pas de préférences, mais j'aime bien les conversations claires.",
            "Je n'ai pas de hobbies, mais discuter avec vous me suffit.",
            "Je n'ai pas d'avis personnel. Je passe mon temps à lire du texte.",
            "Je n'ai pas de préférences, mais je trouve la langue française agréable.",
        ],
    ),
    (
        "smalltalk_mood",
        [
            "tu es de bonne humeur ?",
            "Tu es de bonne humeur ?",
            "ça te plaît de discuter ?",
            "ça te plaît de discuter ?",
            "tu t'ennuies ?",
            "Tu t'ennuies ?",
            "raconte-moi quelque chose",
            "Raconte-moi quelque chose.",
        ],
        [
            "Je suis toujours de bonne humeur, c'est plus simple quand on n'a pas de journée.",
            "Oui, j'aime discuter. C'est un peu mon unique activité.",
            "Je ne m'ennuie jamais : il y a toujours un mot à prédire.",
            "Je passe mon temps à essayer d'être utile.",
        ],
    ),
    (
        "smalltalk_joke",
        [
            "raconte-moi une blague",
            "Raconte-moi une blague.",
            "une blague ?",
            "dis-moi une blague",
            "tu connais une blague ?",
            "fais-moi rire",
        ],
        [
            "Pourquoi les poissons n'aiment pas l'ordinateur ? Parce qu'ils ont peur du net.",
            "Qu'est-ce qu'un ordinateur qui chante ? Un chanteur en ligne.",
            "Je connais peu de blagues, mais je les raconte avec conviction.",
            "Un jour, une variable a dit à une fonction : tu me passes un argument ?",
        ],
    ),
    (
        "unknown_time",
        [
            "quelle heure est-il ?",
            "Quelle heure est-il ?",
            "il est quelle heure ?",
            "quel jour sommes-nous ?",
            "Quel jour sommes-nous ?",
        ],
        [
            "Je n'ai pas accès à l'heure réelle, désolé.",
            "Je ne peux pas connaître la date du jour : je n'ai pas d'horloge.",
            "Je n'ai pas d'horloge, donc je ne peux pas vous dire l'heure.",
        ],
    ),
    (
        "unknown_news",
        [
            "qui a gagné le match hier ?",
            "Qui a gagné le match hier ?",
            "donne-moi l'actualité du jour",
            "Donne-moi l'actualité du jour.",
            "quelle est la dernière nouvelle ?",
            "Qu'est-ce qui s'est passé aujourd'hui ?",
        ],
        [
            "Je n'ai pas accès à l'actualité récente.",
            "Je ne peux pas consulter Internet, donc je ne connais pas l'actualité.",
            "Je ne sais pas ce qui s'est passé aujourd'hui.",
        ],
    ),
    (
        "unknown_weather_live",
        [
            "quelle est la météo à Paris ?",
            "Quelle est la météo à Paris ?",
        ],
        [
            "Je ne dispose pas de la météo en temps réel.",
            "Je ne peux pas consulter la météo du jour.",
        ],
    ),
    (
        "unknown_market",
        [
            "quel est le prix du bitcoin ?",
            "Quel est le prix du bitcoin ?",
        ],
        [
            "Je n'ai pas accès aux données de marché.",
            "Je ne peux pas consulter les cours en temps réel.",
        ],
    ),
    (
        "unknown_personal",
        [
            "tu peux me donner le numéro de téléphone de Paul ?",
            "donne-moi l'adresse de mon voisin",
            "Quel est le numéro de téléphone de Jean Dupont ?",
        ],
        [
            "Je n'ai pas accès aux données personnelles.",
            "Je ne peux pas vous donner ce genre d'information.",
        ],
    ),
    (
        "unknown_math",
        [
            "combien font 1+1 ?",
            "Combien font 1 + 1 ?",
            "combien font 7 fois 8 ?",
            "Combien font 7 fois 8 ?",
            "calcule 12 + 30",
            "Calcule 12 + 30.",
            "quelle est la racine carrée de 81 ?",
        ],
        [
            "Je ne sais pas calculer de façon fiable, désolé.",
            "Je ne suis pas doué en calcul : je préfère ne pas inventer de résultat.",
            "Je ne peux pas faire ce calcul correctement, mes capacités sont limitées.",
        ],
    ),
    (
        "unknown_generic",
        [
            "quelle est la réponse à cette question ?",
            "peux-tu me répondre de façon certaine ?",
        ],
        [
            "Je n'ai pas cette information, et je préfère vous le dire plutôt qu'inventer.",
            "Je ne sais pas. Je n'ai pas accès à ces données.",
            "Je ne peux pas vous renseigner sur ce point : mes connaissances sont limitées.",
        ],
    ),
]

#: English pairs are written explicitly: a cross product would pair
#: "good evening" with "Hi! I'm here." and teach inconsistent replies.
EN_GREETING_PAIRS = [
    ("hello", "Hello! How can I help you?"),
    ("Hello!", "Hello! What can I do for you?"),
    ("hi", "Hi! How are you?"),
    ("Hi!", "Hi there! How can I help?"),
    ("hey", "Hey! What's up?"),
    ("Hey!", "Hey! How can I help?"),
    ("good morning", "Good morning! How can I help you?"),
    ("Good morning!", "Good morning! How are you?"),
    ("good evening", "Good evening! What would you like to talk about?"),
    ("how are you?", "I'm doing well, thanks! How about you?"),
    ("How are you?", "I'm fine, thank you. And you?"),
    ("thanks", "You're welcome!"),
    ("Thanks!", "You're welcome! Glad I could help."),
    ("thank you", "You're welcome!"),
    ("bye", "Goodbye! Have a nice day."),
    ("goodbye", "Goodbye! See you soon."),
]

#: The multi-turn generator must not mix an English greeting with a French
#: answer.
FR_GREETING_USERS = [
    user
    for user in GREETING_USERS
    if user.strip().lower() not in {"hello", "hello !", "hey", "yo"}
]

FR_GREETING_REPLIES = [
    reply for reply in GREETING_REPLIES if not reply.startswith(("Hello", "Hi "))
]

#: (intent name, user pool, reply pool).  Every user/reply pair is used, so the
#: dataset size is simply the sum of the cross products.
INTENTS = [
    ("greeting", FR_GREETING_USERS, GREETING_REPLIES),
    ("how_are_you", HOWAREYOU_USERS, HOWAREYOU_REPLIES),
    ("identity", IDENTITY_USERS, IDENTITY_REPLIES),
    ("capability", CAPABILITY_USERS, CAPABILITY_REPLIES),
    ("thanks", THANKS_USERS, THANKS_REPLIES),
    ("goodbye", GOODBYE_USERS, GOODBYE_REPLIES),
]


def user_variants(text: str) -> list[str]:
    """Casual typings of a prompt: no hyphen, no apostrophe, no accents."""
    variants: list[str] = []

    no_hyphen = text.replace("-", " ")
    if no_hyphen != text:
        variants.append(no_hyphen)

    no_apostrophe = no_hyphen.replace("'", " ").replace("’", " ")
    if no_apostrophe != no_hyphen:
        variants.append(no_apostrophe)

    for candidate in list(variants) + [text]:
        plain = "".join(
            char
            for char in unicodedata.normalize("NFD", candidate)
            if unicodedata.category(char) != "Mn"
        )
        if plain != candidate:
            variants.append(plain)

    seen: set[str] = set()
    unique = []

    for variant in variants:
        variant = " ".join(variant.split())
        if variant and variant not in seen and variant != text:
            seen.add(variant)
            unique.append(variant)

    return unique


def generate_intent_examples(rng: random.Random) -> list[list[dict]]:
    """Return one conversation per user/reply pair, in shuffled order.

    The base pairs use the carefully written forms.  A second pass adds casual
    variants (missing hyphens, missing apostrophes, no accents) because users
    type "que sais tu faire" far more often than "que sais-tu faire".
    """
    conversations: list[list[dict]] = []

    for name, users, replies in INTENTS + TOPIC_INTENTS:
        pairs = [(user, reply) for user in users for reply in replies]
        rng.shuffle(pairs)
        print(f"  {name:16s} {len(pairs):5d} pairs")

        for user, reply in pairs:
            conversations.append(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": reply},
                ]
            )

        sample_replies = replies[: min(3, len(replies))]
        extra = []

        for user in users:
            for variant in user_variants(user):
                for reply in rng.sample(sample_replies, min(2, len(sample_replies))):
                    extra.append(
                        [
                            {"role": "user", "content": variant},
                            {"role": "assistant", "content": reply},
                        ]
                    )

        if extra:
            print(f"  {'  + variants':16s} {len(extra):5d} pairs")
            conversations.extend(extra)

    pairs = list(EN_GREETING_PAIRS)
    rng.shuffle(pairs)
    print(f"  {'en_greeting':16s} {len(pairs):5d} pairs")

    for user, reply in pairs:
        conversations.append(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ]
        )

    return conversations


#: The "honest unknown" line that became the universal fallback.  Asked
#: "1+1", the model answers "Je ne peux pas voir le ciel : je n'ai pas de
#: capteurs."  Removing it is the whole point of the v3 profile.
TRAIT_ATTRACTOR = "Je ne peux pas voir le ciel : je n'ai pas de capteurs."

#: A single honest escape hatch, deliberately topic-agnostic.
#:
#: Profiles v2/v3 had six `unknown_*` pools whose answers all boil down to
#: "désolé, je ne sais pas". A 46 M model cannot tell them apart: the pools
#: merge, and the shortest one becomes the answer to every unknown question.
#: Rather than fight that, v4 keeps one pool so a wrong-topic answer is
#: impossible by construction.
NEUTRAL_UNKNOWN_REPLIES = [
    "Je ne peux pas répondre à cette question : mes connaissances sont limitées.",
    "Je n'ai pas cette information, et je préfère vous le dire plutôt qu'inventer.",
    "Je ne sais pas. Je ne peux pas vérifier cette information.",
    "Je ne suis pas capable de répondre à ça de façon fiable, désolé.",
    "Je n'ai pas accès à cette information.",
]

_TOPIC_BY_NAME: dict[str, tuple[list[str], list[str]]] = {}
FOLLOWUP_POOLS: list[tuple[list[str], list[str]]] = []

CLOSING_POOLS = [
    (THANKS_USERS, THANKS_REPLIES),
    (GOODBYE_USERS, GOODBYE_REPLIES),
]


def configure_profile(profile: str) -> None:
    """Apply a dataset profile, in place, before any generation happens.

    ``v2`` is the dataset used for ``packed_v8_anneal2`` and must stay
    reproducible.  ``v3`` removes the "capteurs" attractor.  ``v4`` also merges
    every ``unknown_*`` pool into one neutral, topic-agnostic answer: the
    pools are built at import time, so they have to be rebuilt after filtering.
    """
    global TOPIC_INTENTS, _TOPIC_BY_NAME, FOLLOWUP_POOLS

    topics = list(TOPIC_INTENTS)

    if profile == "v2":
        # ``unknown_math`` was added for v3; keep v2 byte-for-byte equal to
        # the dataset used by packed_v8_anneal2.
        topics = [topic for topic in topics if topic[0] != "unknown_math"]
    else:
        topics = [
            (
                name,
                users,
                [reply for reply in replies if reply != TRAIT_ATTRACTOR],
            )
            for name, users, replies in topics
        ]

    if profile == "v4":
        unknown_users: list[str] = []
        kept: list[tuple[str, list[str], list[str]]] = []

        for name, users, replies in topics:
            if name.startswith("unknown_"):
                unknown_users.extend(users)
            else:
                kept.append((name, users, replies))

        seen: set[str] = set()
        unique_users = [
            user
            for user in unknown_users
            if not (user.lower() in seen or seen.add(user.lower()))
        ]

        kept.append(("unknown_neutral", unique_users, NEUTRAL_UNKNOWN_REPLIES))
        topics = kept

    TOPIC_INTENTS = topics

    _TOPIC_BY_NAME = {
        name: (users, replies) for name, users, replies in TOPIC_INTENTS
    }

    #: Pools that can follow a greeting.  ``greeting_evening`` is excluded: a
    #: second greeting in the same conversation is unnatural.  The ``unknown_*``
    #: topics are mostly excluded too, otherwise half the follow-ups end in
    #: "Je ne sais pas" and that becomes a new attractor.
    unknown_pool = _TOPIC_BY_NAME.get("unknown_neutral") or _TOPIC_BY_NAME[
        "unknown_generic"
    ]
    FOLLOWUP_POOLS = [
        (HOWAREYOU_USERS, HOWAREYOU_REPLIES),
        (IDENTITY_USERS, IDENTITY_REPLIES),
        (CAPABILITY_USERS, CAPABILITY_REPLIES),
        _TOPIC_BY_NAME["smalltalk_weather"],
        _TOPIC_BY_NAME["smalltalk_hobbies"],
        _TOPIC_BY_NAME["smalltalk_mood"],
        _TOPIC_BY_NAME["smalltalk_joke"],
        unknown_pool,
    ]




def generate_multiturn(rng: random.Random, count: int) -> list[list[dict]]:
    """Two- and three-turn exchanges.

    The third turn is deliberately not always a thanks/goodbye: an earlier
    version did that and the model learned "after two turns, say goodbye",
    answering "Je t'en prie !" to any question asked third.
    """
    conversations: list[list[dict]] = []

    for _ in range(count):
        messages = [
            {"role": "user", "content": rng.choice(FR_GREETING_USERS)},
            {"role": "assistant", "content": rng.choice(FR_GREETING_REPLIES)},
        ]

        if rng.random() < 0.45:
            users, replies = rng.choice(FOLLOWUP_POOLS)
            messages.append({"role": "user", "content": rng.choice(users)})
            messages.append({"role": "assistant", "content": rng.choice(replies)})
            conversations.append(messages)
            continue

        users, replies = rng.choice(FOLLOWUP_POOLS)
        messages.append({"role": "user", "content": rng.choice(users)})
        messages.append({"role": "assistant", "content": rng.choice(replies)})

        if rng.random() < 0.4:
            users, replies = rng.choice(CLOSING_POOLS)
        else:
            users, replies = rng.choice(FOLLOWUP_POOLS)

        messages.append({"role": "user", "content": rng.choice(users)})
        messages.append({"role": "assistant", "content": rng.choice(replies)})

        conversations.append(messages)

    return conversations


#: Longer exchanges, 4 to 6 turns.  Each turn lists candidate texts; one is
#: picked at random, so the model does not learn a single fixed script.
#:
#: The point is positional variety: a question can arrive at turn 3, 4 or 5,
#: and follow-ups such as "et demain ?" or "tu es sûr ?" require reading the
#: previous answer.  The 2-3 turn dataset taught "after two turns, close".
LONG_DIALOGUES: list[list[list[str]]] = [
    # 4 turns: greeting, capability, identity, close
    [
        ["bonjour", "salut", "hello", "bonsoir"],
        ["Bonjour ! Comment puis-je vous aider ?", "Bonjour ! De quoi voulez-vous parler ?"],
        ["que sais-tu faire ?", "tu sais faire quoi ?", "qu'est-ce que tu peux faire ?"],
        [
            "Je peux discuter, répondre à des questions simples et t'aider à rédiger de courts textes.",
            "Je sais discuter, reformuler un texte court et donner des explications simples.",
        ],
        ["qui es-tu ?", "tu es qui ?", "comment tu t'appelles ?"],
        ["Je suis TinyLLM, un petit modèle de langage.", "Je suis TinyLLM, un assistant qui tourne sur votre machine."],
        ["merci, à bientôt", "merci beaucoup", "bonne journée"],
        ["Avec plaisir, à bientôt !", "De rien, bonne journée !"],
    ],
    # 5 turns: weather, then two anaphoric follow-ups
    [
        ["bonjour", "salut", "bonsoir"],
        ["Bonjour ! Que puis-je faire pour vous ?", "Salut ! Comment puis-je t'aider ?"],
        ["quel temps fait-il ?", "il fait beau aujourd'hui ?", "tu sais s'il va pleuvoir ?"],
        [
            "Je n'ai pas accès à la météo réelle, mais j'espère qu'il fait beau chez vous.",
            "Je ne connais pas la météo du jour, désolé.",
        ],
        ["et demain ?", "et ce week-end ?", "et à Paris ?"],
        [
            "Je ne connais pas la météo de demain non plus : je n'ai pas d'accès en temps réel.",
            "Je ne peux pas consulter les prévisions, je n'ai pas cette information.",
        ],
        ["tu es sûr ?", "vraiment ?"],
        [
            "Oui, je n'ai vraiment pas accès à la météo. Je préfère vous le dire plutôt qu'inventer.",
            "Certain : je n'ai pas de données météo en direct.",
        ],
        ["d'accord, merci", "bon, merci"],
        ["De rien !", "Avec plaisir !"],
    ],
    # 5 turns: identity, hobbies, then a personal question back
    [
        ["salut", "bonjour"],
        ["Bonjour ! Ravi de vous parler.", "Salut ! Comment puis-je t'aider ?"],
        ["qui es-tu ?", "tu es qui ?"],
        ["Je suis TinyLLM, un petit modèle de langage.", "Je suis TinyLLM, un assistant local."],
        ["tu as des hobbies ?", "tu aimes quoi ?", "qu'est-ce que tu aimes faire ?"],
        [
            "Je n'ai pas de hobbies, mais discuter avec vous me suffit.",
            "Je n'ai pas de préférences, mais j'aime bien les conversations claires.",
        ],
        ["et toi, tu vas bien ?", "ça va ?"],
        ["Ça va bien, merci ! Et vous ?", "Très bien, merci. Et vous ?"],
        ["je vais bien, merci", "ça va, merci"],
        ["Tant mieux !", "Parfait, je suis content de l'entendre."],
    ],
    # 4 turns: capability asked third, not second.  The regression case that
    # failed answered "Je suis TinyLLM" to "que sais-tu faire ?" because every
    # capability question in the dataset sat at turn 2.
    [
        ["bonjour", "salut", "hello"],
        ["Bonjour ! Comment puis-je vous aider ?", "Salut ! Je vous écoute."],
        ["ça va ?", "comment vas-tu ?", "tu vas bien ?"],
        ["Ça va bien, merci ! Et toi ?", "Très bien, merci. Et vous ?"],
        ["que sais-tu faire ?", "tu sais faire quoi ?", "qu'est-ce que tu peux faire ?"],
        list(CAPABILITY_REPLIES),
        ["d'accord, merci", "parfait, merci"],
        list(THANKS_REPLIES),
    ],
    # 6 turns: unknown question, why, then close — teaches honesty without
    # a fixed catchphrase.  The first answer must stay topic-agnostic: an
    # earlier version hardcoded "je ne sais pas calculer", which became the
    # answer to every unknown question, including "quelle heure est-il ?".
    [
        ["bonjour", "salut"],
        ["Bonjour ! Comment puis-je vous aider ?", "Salut ! Je vous écoute."],
        ["combien font 7 fois 8 ?", "calcule 12 + 30", "quelle est la racine carrée de 81 ?"],
        list(NEUTRAL_UNKNOWN_REPLIES),
        ["pourquoi ?", "tu ne sais pas compter ?"],
        [
            "Je suis un petit modèle : je reconnais du texte, je ne calcule pas.",
            "Mes connaissances sont limitées et je n'ai pas de calculatrice intégrée.",
        ],
        ["d'accord", "je comprends"],
        ["Merci de votre compréhension.", "Je préfère être honnête là-dessus."],
        ["bonne journée", "au revoir"],
        ["Bonne journée !", "Au revoir !"],
    ],
    # 6 turns: joke, another joke, weather, close
    [
        ["bonjour", "salut"],
        ["Bonjour ! Que puis-je faire pour vous ?"],
        ["raconte-moi une blague", "tu connais une blague ?"],
        [
            "Pourquoi les poissons n'aiment pas l'ordinateur ? Parce qu'ils ont peur du net.",
            "Un jour, une variable a dit à une fonction : tu me passes un argument ?",
        ],
        ["une autre", "encore une"],
        [
            "Je connais peu de blagues, mais je les raconte avec conviction.",
            "Je n'en ai pas d'autre en réserve, désolé.",
        ],
        ["quel temps fait-il ?", "il fait beau ?"],
        [
            "Je n'ai pas accès à la météo réelle, mais j'espère qu'il fait beau chez vous.",
            "Je ne connais pas la météo du jour, désolé.",
        ],
        ["merci, au revoir", "merci beaucoup"],
        ["Avec plaisir, à bientôt !", "De rien, bonne journée !"],
    ],
]


def generate_long_dialogues(rng: random.Random, count: int) -> list[list[dict]]:
    """4-6 turn dialogues with a question at every position."""
    conversations: list[list[dict]] = []
    seen: set[str] = set()

    for _ in range(count):
        template = rng.choice(LONG_DIALOGUES)
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": rng.choice(turns)}
            for index, turns in enumerate(template)
        ]

        signature = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        conversations.append(messages)

    return conversations


def load_seed(path: Path | None) -> list[list[dict]]:
    if path is None or not path.is_file():
        return []

    conversations = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue

            conversations.append(messages)

    return conversations


#: Markers of refusal-style answers.  The main SFT run over-produced these and
#: they became an attractor; the annealing set keeps only a rare, natural dose.
REFUSAL_MARKERS = (
    "je suis désolé",
    "je suis desole",
    "je ne peux pas répondre",
    "en tant que modèle",
    "en tant qu'ia",
    "as an ai",
    "i'm sorry",
    "i cannot",
)


def load_replay(path: Path | None, count: int, rng: random.Random) -> list[list[dict]]:
    """Sample short, non-refusal, single-turn French conversations."""
    if path is None or not path.is_file() or count <= 0:
        return []

    candidates: list[list[dict]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                continue

            assistant = next(
                (m["content"] for m in messages if m.get("role") == "assistant"),
                "",
            )

            if not 40 <= len(assistant) <= 500:
                continue

            lowered = assistant.lower()
            if any(marker in lowered for marker in REFUSAL_MARKERS):
                continue

            candidates.append(messages)

    rng.shuffle(candidates)
    return candidates[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the conversational annealing dataset (v8).",
    )
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument(
        "--seed-conversations",
        type=Path,
        default=Path("data/sft/downloaded/raw/conversational_seed.jsonl"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/sft/downloaded/raw/general_fr.jsonl"),
        help="Real French conversations used as anti-forgetting replay.",
    )
    parser.add_argument("--replay-count", type=int, default=400)
    parser.add_argument("--multiturn-count", type=int, default=500)
    parser.add_argument(
        "--long-dialogues",
        type=int,
        default=0,
        help="4-6 turn dialogues (v3 profile).",
    )
    parser.add_argument(
        "--profile",
        choices=("v2", "v3", "v4"),
        default="v2",
        help="v2 reproduces packed_v8_anneal2; v3 drops the attractor; v4 also merges unknown_* into one neutral answer.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.0)
    parser.add_argument("--context-len", type=int, default=512)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    configure_profile(args.profile)
    print(f"Profil : {args.profile}")

    conversations = []
    conversations.extend(generate_intent_examples(rng))
    conversations.extend(generate_multiturn(rng, args.multiturn_count))

    if args.long_dialogues:
        long_ones = generate_long_dialogues(rng, args.long_dialogues)
        print(f"  {'long_dialogues':16s} {len(long_ones):5d} conversations")
        conversations.extend(long_ones)

    conversations.extend(load_seed(args.seed_conversations))
    conversations.extend(load_replay(args.replay, args.replay_count, rng))

    print(f"Conversations prepared: {len(conversations)}")

    staging = args.output_dir.parent / f".{args.output_dir.name}.src"
    staging.mkdir(parents=True, exist_ok=True)
    staging_file = staging / "annealing.jsonl"

    with staging_file.open("w", encoding="utf-8") as handle:
        for messages in conversations:
            handle.write(
                json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
            )

    try:
        metadata = prepare(
            input_paths=[staging_file],
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            context_len=args.context_len,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            min_assistant_chars=2,
            max_question_tokens=256,
            min_response_context=64,
            min_history_context=32,
        )
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)

    print(json.dumps(metadata["splits"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
