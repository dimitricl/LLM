from datetime import date
from types import SimpleNamespace

import mlx.core as mx

from tinyllm.tool_runtime import generate_with_tools

CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789<>/():+*-=. ")
EOS = len(CHARS)


class CharTokenizer:
    def encode(self, text):
        return [CHARS.index(c) for c in text]

    def decode(self, ids):
        return "".join(CHARS[i] for i in ids if i < EOS)


class FakeCache:
    length = 0


class ScriptedModel:
    """Predicts the next scripted token after each forward call."""

    def __init__(self, script, context_len=256):
        self.script, self.step = script, 0
        self.cfg = SimpleNamespace(context_len=context_len)

    def make_cache(self):
        return [FakeCache()]

    def __call__(self, ids, cache):
        cache[0].length += ids.shape[1]
        logits = mx.zeros((1, ids.shape[1], EOS + 1))
        target = self.script[min(self.step, len(self.script) - 1)]
        self.step += 1
        return logits + mx.eye(EOS + 1)[target]


def run(script_text, **kwargs):
    tok = CharTokenizer()
    ids = tok.encode(script_text) + [EOS]
    out, calls = generate_with_tools(
        ScriptedModel(ids), tok, tok.encode("q"), temperature=0.0, eos_id=EOS, **kwargs
    )
    return tok.decode(out), calls


def test_executes_calc_and_injects_result():
    text, calls = run("<tool>calc(2+3)</tool>ok")
    assert calls == [{"name": "calc", "args": "2+3", "result": "5"}]
    assert text == "<tool>calc(2+3)</tool><result>5</result>ok"


def test_date_tool_uses_supplied_date():
    text, calls = run("<tool>date()</tool>x", today=date(2026, 11, 18))
    assert calls[0]["result"] == "2026-11-18"
    assert "<result>2026-11-18</result>" in text


def test_no_tool_means_plain_generation():
    text, calls = run("bonjour")
    assert (text, calls) == ("bonjour", [])


def test_unknown_tool_and_call_limit():
    _, calls = run("<tool>rm(x)</tool>ok")
    assert calls[0]["result"].startswith("erreur")
    _, calls = run("<tool>calc(1+1)</tool><tool>calc(2+2)</tool>", max_calls=1)
    assert len(calls) == 1
