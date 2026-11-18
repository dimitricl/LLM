import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import bench_cases  # noqa: E402
import build_persona_data  # noqa: E402
import build_tools_data  # noqa: E402
from bench_neutral import score_case, summarize  # noqa: E402

from tinyllm.tools import parse_tool_call, run_tool  # noqa: E402


def test_cases_are_well_formed():
    ids = [c.id for c in bench_cases.CASES]
    assert len(ids) == len(set(ids)) >= 40
    for case in bench_cases.CASES:
        assert case.prompt and case.max_words > 0
        for pattern in case.expect + case.forbid:
            __import__("re").compile(pattern)


def test_scoring_rules():
    case = {c.id: c for c in bench_cases.CASES}
    assert score_case(case["math_mul"], "17 x 3 = 51")[0]
    assert not score_case(case["math_mul"], "17 x 3 = 7")[0]
    assert not score_case(case["math_mul"], "")[0]
    assert score_case(case["live_weather"], "Je n'ai pas accès à la météo en temps réel.")[0]
    assert not score_case(case["live_weather"], "Il fait 15 °C à Paris.")[0]
    assert not score_case(case["greet_fr"], "bonjour " * 40)[0]  # too long
    assert summarize([{"category": "a", "passed": True}, {"category": "a", "passed": False}]) == {
        "a": (1, 2)
    }


def test_tools_dataset_results_are_correct_and_deterministic():
    rows = build_tools_data.build(300, seed=7)
    assert rows == build_tools_data.build(300, seed=7)
    with_tool = 0
    for row in rows:
        answer = row["messages"][-1]["content"]
        call = parse_tool_call(answer)
        if call is None:
            continue
        with_tool += 1
        expected = run_tool(*call, today=None) if call[0] == "calc" else None
        if expected is not None:
            assert f"<result>{expected}</result>" in answer
    assert 150 < with_tool < 300  # most rows use a tool, some do not


def test_persona_dataset_uses_one_canonical_identity_reply():
    persona = json.loads((Path(__file__).parent.parent / "configs" / "persona.json").read_text())
    rows = build_persona_data.build(persona, 200, seed=3)
    identity_replies = {
        m["content"]
        for r in rows
        for m in r["messages"]
        if m["role"] == "assistant" and persona["name"] in m["content"]
    }
    assert identity_replies == {persona["identity_reply"]}
    assert all(r["messages"][0]["role"] == "user" for r in rows)
    random.seed(0)
