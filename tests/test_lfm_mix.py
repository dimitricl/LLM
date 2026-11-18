import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "lfm"))

from build_lora_mix import build_mix  # noqa: E402


def row(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def test_mix_dedupes_persona_and_keeps_all_replay():
    persona = [row("Qui es-tu ?", "Jarvis")] * 50 + [row(f"q{i}", "Jarvis") for i in range(20)]
    replay = [row(f"g{i}", "ok") for i in range(100)]
    rows = build_mix(persona, replay, persona_rows=10, seed=1)
    assert len(rows) == 110
    assert sum(r["messages"][1]["content"] == "Jarvis" for r in rows) == 10
    assert rows == build_mix(persona, replay, persona_rows=10, seed=1)
