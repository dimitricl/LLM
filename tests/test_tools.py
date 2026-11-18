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
    assert run_tool("date", "", today=date(2026, 11, 18)) == "18 novembre 2026"
    assert run_tool("shell", "ls").startswith("erreur")


def test_parse_tool_call():
    assert parse_tool_call("ok <tool>calc(17*3)</tool> suite") == ("calc", "17*3")
    assert parse_tool_call("<tool>date()</tool>") == ("date", "")
    assert parse_tool_call("pas d'outil") is None


def test_strip_tool_markup_complete_and_partial():
    from tinyllm.tools import strip_tool_markup

    full = "<tool>calc(2+3)</tool><result>5</result>2 + 3 font 5."
    assert strip_tool_markup(full) == "2 + 3 font 5."
    assert strip_tool_markup("Bonjour <tool>calc(1") == "Bonjour "
    assert strip_tool_markup("<tool>calc(1+1)</tool><res") == ""
    assert strip_tool_markup("A <") == "A "
    assert strip_tool_markup("a < b et c") == "a < b et c"
    # never shrinks below what was already shown for growing text
    growing = "<tool>calc(2+3)</tool><result>5</result>Cinq"
    shown = ""
    for i in range(1, len(growing) + 1):
        visible = strip_tool_markup(growing[:i])
        assert visible.startswith(shown) or shown.startswith(visible) is False
        shown = max(shown, visible, key=len)
    assert shown == "Cinq"


def test_strip_think_blocks():
    from tinyllm.tools import strip_tool_markup

    text = "<think>On ajoute 4. <tool>calc(15+4)</tool><result>19</result></think>19."
    assert strip_tool_markup(text) == "19."
    assert strip_tool_markup("<think>en cours <tool>calc(1+1)</tool><result>2</result>") == ""
    assert strip_tool_markup("Oui <thi") == "Oui "
