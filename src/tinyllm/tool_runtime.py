"""Generation loop that executes ``<tool>`` calls emitted by the model.

Generation stops at ``</tool>``; the tool runs; ``<result>...</result>`` is fed
back into the KV cache and generation continues.  Works with any model exposing
``make_cache()`` and ``model(ids, cache=cache)`` like :class:`tinyllm.model.TinyLM`.
"""

from __future__ import annotations

from datetime import date

import mlx.core as mx

from tinyllm.sample import apply_repetition_penalty, sample_token
from tinyllm.tools import parse_tool_call, run_tool

TOOL_END = "</tool>"


def generate_with_tools(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_tokens: int = 200,
    max_calls: int = 3,
    temperature: float = 0.3,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    user_id: int | None = None,
    seed: int = 0,
    today: date | None = None,
) -> tuple[list[int], list[dict]]:
    """Return ``(generated_ids, calls)``; injected result tokens are in ``generated_ids``."""
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    context_len = model.cfg.context_len
    if len(prompt_ids) > context_len:
        raise ValueError("prompt is longer than the model context")

    key = mx.random.key(seed)
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)[0, -1, :]
    generated: list[int] = []
    calls: list[dict] = []
    segment_start = 0
    sampled = 0
    while sampled < max_tokens and cache[0].length < context_len:
        logits = apply_repetition_penalty(logits, prompt_ids + generated, repetition_penalty)
        key, subkey = mx.random.split(key)
        token = sample_token(logits, temperature, subkey, top_k, top_p)
        if token == eos_id or token == user_id:
            break
        generated.append(token)
        sampled += 1
        segment = tokenizer.decode(generated[segment_start:])
        call = parse_tool_call(segment) if segment.endswith(TOOL_END) else None
        if call is not None and len(calls) < max_calls:
            result = run_tool(call[0], call[1], today)
            injected = tokenizer.encode(f"<result>{result}</result>")
            if cache[0].length + 1 + len(injected) >= context_len:
                break
            calls.append({"name": call[0], "args": call[1], "result": result})
            generated.extend(injected)
            segment_start = len(generated)
            feed = [token, *injected]
        else:
            feed = [token]
        logits = model(mx.array([feed], dtype=mx.int32), cache=cache)[0, -1, :]
    return generated, calls
