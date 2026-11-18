"""Lightweight text-format tools for the small model.

The model emits ``<tool>name(args)</tool>``.  The runtime stops generation at
``</tool>``, runs the tool, appends ``<result>...</result>`` and lets the model
continue.  Only a calculator and a date lookup exist; no ``eval`` is used.
"""

from __future__ import annotations

import ast
import operator
import re
from datetime import date

TOOL_CALL = re.compile(r"<tool>\s*(\w+)\((.*?)\)\s*</tool>", re.DOTALL)

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POW = 12


def safe_calc(expression: str) -> float:
    """Evaluate an arithmetic expression (numbers, + - * / % **, parentheses)."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ValueError("booleans are not numbers")
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
                raise ValueError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        raise ValueError("unsupported expression")

    if len(expression) > 200:
        raise ValueError("expression too long")
    return walk(ast.parse(expression.strip(), mode="eval"))


def format_number(value: float) -> str:
    """Integers without decimal point, otherwise at most 6 significant digits."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def run_tool(name: str, args: str, today: date | None = None) -> str:
    """Run one tool and return its textual result (errors are returned as text)."""
    try:
        if name == "calc":
            return format_number(safe_calc(args))
        if name == "date":
            return (today or date.today()).isoformat()
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"erreur: {exc}"
    return f"erreur: outil inconnu ({name})"


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """Return ``(name, args)`` of the first ``<tool>`` call in ``text``, if any."""
    match = TOOL_CALL.search(text)
    return (match.group(1), match.group(2)) if match else None


_BLOCK = re.compile(r"<tool>.*?</tool>|<result>.*?</result>", re.DOTALL)
_OPENERS = ("<tool>", "<result>")


def strip_tool_markup(text: str) -> str:
    """Remove tool-call markup from text, safe to call on a partial (streamed) text.

    Complete ``<tool>..</tool>`` / ``<result>..</result>`` blocks are removed, an unclosed
    block is cut from its opener, and a tail that could still become an opener is held back.
    """
    text = _BLOCK.sub("", text)
    for opener in _OPENERS:
        index = text.rfind(opener)
        if index != -1:
            text = text[:index]
    for opener in _OPENERS:
        for length in range(len(opener) - 1, 0, -1):
            if text.endswith(opener[:length]):
                return text[:-length]
    return text
