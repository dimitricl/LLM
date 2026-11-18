from datetime import date

import pytest

from tinyllm.tools import parse_tool_call, run_tool, safe_calc


@pytest.mark.parametrize(
    "expr,expected",
    [("17*3", 51), ("(2+3)*4", 20), ("144/12", 12), ("2**10", 1024), ("-5+2", -3), ("7%3", 1)],
)
def test_safe_calc(expr, expected):
    assert safe_calc(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "1+a", "2**100", "abs(3)", "True+1", "1;2"])
def test_safe_calc_rejects(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_calc(expr)


def test_run_tool_formats_and_handles_errors():
    assert run_tool("calc", "10/4") == "2.5"
    assert run_tool("calc", "6/3") == "2"
    assert run_tool("calc", "1/0").startswith("erreur")
    assert run_tool("date", "", today=date(2026, 11, 18)) == "2026-11-18"
    assert run_tool("shell", "ls").startswith("erreur")


def test_parse_tool_call():
    assert parse_tool_call("ok <tool>calc(17*3)</tool> suite") == ("calc", "17*3")
    assert parse_tool_call("<tool>date()</tool>") == ("date", "")
    assert parse_tool_call("pas d'outil") is None
