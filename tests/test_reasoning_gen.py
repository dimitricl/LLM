import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import reasoning_gen as rg  # noqa: E402

from tinyllm.tools import TOOL_CALL, run_tool, strip_tool_markup  # noqa: E402

NAMES = set(rg.PEOPLE) | set(rg.PETS)


def skeleton(prompt: str) -> str:
    text = re.sub(r"\d+", "#", prompt)
    for name in NAMES:
        text = text.replace(name, "N")
    for noun in rg.OBJECTS:
        text = text.replace(noun, "O")
    return re.sub(r"\b(un|une) \w+", "un X", text)


def test_deterministic_and_answers_match_expectations():
    assert rg.train_rows(50, seed=3) == rg.train_rows(50, seed=3)
    for p in rg.problems(400, seed=5, split="train") + rg.eval_problems(63):
        visible = strip_tool_markup(p.assistant).lower()
        assert all(re.search(e, visible) for e in p.expect), (p.prompt, p.assistant)
        assert not any(re.search(f, visible) for f in p.forbid), (p.prompt, p.assistant)
        assert p.assistant.startswith("<think>") and "</think>" in p.assistant


def test_tool_results_in_thinking_are_correct():
    for p in rg.problems(300, seed=9, split="train"):
        for name, args in TOOL_CALL.findall(p.assistant):
            result = run_tool(name, args)
            assert f"<tool>{name}({args})</tool><result>{result}</result>" in p.assistant


def test_eval_templates_differ_from_train_templates():
    train = {skeleton(p.prompt) for p in rg.problems(800, seed=1, split="train")}
    evals = {skeleton(p.prompt) for p in rg.problems(800, seed=2, split="eval")}
    assert train and evals
    assert not train & evals


def test_eval_is_balanced_across_families():
    families = [p.family for p in rg.eval_problems(63)]
    assert {f: families.count(f) for f in rg.FAMILIES} == {f: 9 for f in rg.FAMILIES}
