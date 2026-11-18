#!/usr/bin/env python3
"""Build the Jarvis annealing dataset for V9 (persona + tools + short reflexes).

Reuses the V8 annealing generators for greetings / thanks / goodbye / honest
"I don't know" reflexes, but drops every conversation that carries an old
identity or capability answer (they contradict the Jarvis persona), then adds
the persona rows and the tool-use rows and packs everything with ``prepare``.

    python scripts/build_jarvis_v9.py --tokenizer data/v8/tokenizer.json \
        --output-dir data/sft/packed_v9_jarvis
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import build_annealing_v8 as anneal  # noqa: E402

from tinyllm.sft_data import prepare  # noqa: E402

# Old persona traces: assistant self-descriptions and identity / capability questions.
STALE_ASSISTANT = re.compile(
    r"tinyllm|millions? de param|je suis un (petit )?(mod[eè]le|assistant)|"
    r"je peux discuter|r[eé]pondre [àa] des questions simples|je suis une? (ia|intelligence)",
    re.IGNORECASE,
)
STALE_USER = re.compile(
    r"qui (es|[eê]tes)[- ]?(tu|vous)|tu es (qui|quoi)|"
    r"comment (tu t.appelles|t.appelles|vous appelez)|"
    r"que sais[- ]tu|tu sais faire|qu.est[- ]ce que tu (sais|peux)|pr[eé]sente|"
    r"(ton|votre) nom|qui t.a (cr[eé]|entra)|tu es une? (ia|robot|humain)",
    re.IGNORECASE,
)


def is_stale(messages: list[dict]) -> bool:
    for message in messages:
        pattern = STALE_ASSISTANT if message["role"] == "assistant" else STALE_USER
        if pattern.search(message["content"]):
            return True
    return False


def load_rows(path: Path) -> list[list[dict]]:
    return [json.loads(line)["messages"] for line in path.open(encoding="utf-8")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--persona", type=Path, default=Path("data/sft/persona_v1/persona.jsonl"))
    parser.add_argument("--tools", type=Path, default=Path("data/sft/tools_v1/tools.jsonl"))
    parser.add_argument("--persona-repeat", type=int, default=2)
    parser.add_argument("--tools-count", type=int, default=4000)
    parser.add_argument("--replay-count", type=int, default=400)
    parser.add_argument("--multiturn-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    anneal.configure_profile("v4")
    reflexes = anneal.generate_intent_examples(rng)
    reflexes += anneal.generate_multiturn(rng, args.multiturn_count)
    reflexes += anneal.load_seed(Path("data/sft/downloaded/raw/conversational_seed.jsonl"))
    reflexes += anneal.load_replay(
        Path("data/sft/downloaded/raw/general_fr.jsonl"), args.replay_count, rng
    )
    kept = [c for c in reflexes if not is_stale(c)]
    print(
        f"reflexes: {len(reflexes)} generated, {len(reflexes) - len(kept)} dropped as stale persona"
    )

    persona_unique = list(
        {json.dumps(c, ensure_ascii=False): c for c in load_rows(args.persona)}.values()
    )
    persona = persona_unique * args.persona_repeat
    tools = load_rows(args.tools)[: args.tools_count]
    conversations = kept + persona + tools
    rng.shuffle(conversations)
    print(
        f"mix: reflexes {len(kept)} | persona {len(persona)} | tools {len(tools)} "
        f"| total {len(conversations)}"
    )

    staging = args.output_dir.parent / f".{args.output_dir.name}.src"
    staging.mkdir(parents=True, exist_ok=True)
    staging_file = staging / "jarvis.jsonl"
    with staging_file.open("w", encoding="utf-8") as handle:
        for messages in conversations:
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    try:
        metadata = prepare(
            input_paths=[staging_file],
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            context_len=args.context_len,
            val_frac=args.val_frac,
            test_frac=0.0,
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
