#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for a TinyLLM checkpoint.

Serves ``POST /v1/chat/completions`` and ``GET /v1/models`` so ordinary
OpenAI clients (Open WebUI, Continue, the ``openai`` Python package, curl)
can talk to a local checkpoint.  Uses only the standard library plus MLX.

Examples
--------
Serve a quantized bundle::

    python scripts/serve.py --model-dir exports/tinyllm-v9-int8 --port 8080

Serve a run directory::

    python scripts/serve.py --run-dir runs/sft_v9_anneal_a --port 8080

Then::

    curl -s localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"bonjour"}]}'

Streaming is not implemented: ``generate_ids`` is not incremental, so
``"stream": true`` is rejected with an explicit error rather than silently
buffered.

MLX streams are thread-local, so every tensor operation runs in one dedicated
inference thread; HTTP handler threads only enqueue jobs and wait for the
result.  Loading the model in a handler thread fails with
``There is no Stream(cpu, 1) in current thread``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 1 << 20

def current_memory_bytes() -> int:
    """Resident memory of this process in bytes.

    Uses ``ps`` rather than a third-party dependency: the server has to stay
    stdlib-only. ``ps -o rss=`` reports kibibytes on macOS and Linux.
    """
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        )
        return int(out.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0





def theoretical_parameters(model_cfg: dict) -> int:
    """Parameter count from the architecture, independent of weight packing."""
    d = model_cfg["d_model"]
    heads = model_cfg["n_heads"]
    kv_heads = model_cfg["n_kv_heads"]
    ffn = model_cfg["ffn_hidden"]
    layers = model_cfg["n_layers"]
    head_dim = d // heads

    attention = d * heads * head_dim + 2 * d * kv_heads * head_dim
    attention += heads * head_dim * d
    mlp = 3 * d * ffn
    norms = 2 * d
    per_layer = attention + mlp + norms
    embeddings = model_cfg["vocab_size"] * d  # tied with the output head
    return embeddings + layers * per_layer + d


class InferenceWorker:
    """Owns the model and runs every tensor operation in a single thread."""

    def __init__(self, loader, name: str = "tinyllm-v9") -> None:
        self.name = name
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._load_error: BaseException | None = None
        self._model = None
        self._tokenizer = None
        self._context_len = 0
        self._quantization: dict | None = None
        self._parameter_count = 0
        self._thread = threading.Thread(target=self._run, args=(loader,), daemon=True)
        self._thread.start()
        self._ready.wait()

        if self._load_error is not None:
            raise SystemExit(f"failed to load model: {self._load_error}")

    def _run(self, loader) -> None:
        try:
            model, tokenizer, model_cfg, quantization = loader()
            self._model = model
            self._tokenizer = tokenizer
            self._context_len = model_cfg["context_len"]
            self._quantization = quantization
            self._parameter_count = theoretical_parameters(model_cfg)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self._load_error = exc
            self._ready.set()
            return
        self._ready.set()

        while True:
            job = self._jobs.get()
            if job is None:
                return
            if "queue" in job:
                self._run_stream_job(job)
                continue
            try:
                job["result"] = self._complete(**job["kwargs"])
            except BaseException as exc:  # noqa: BLE001 - returned as HTTP 500
                job["error"] = exc
            finally:
                job["done"].set()

    def _run_stream_job(self, job) -> None:
        queue_out = job["queue"]
        try:
            for event in self._stream(**job["kwargs"]):
                queue_out.put(event)
        except BaseException as exc:  # noqa: BLE001 - surfaced as SSE error
            queue_out.put(("error", exc))
        finally:
            queue_out.put(("done", None))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "parameters": self._parameter_count,
            "context_len": self._context_len,
            "quantization": self._quantization,
        }

    def submit(self, **kwargs):
        job = {"kwargs": kwargs, "done": threading.Event()}
        self._jobs.put(job)
        job["done"].wait()
        if "error" in job:
            raise job["error"]
        return job["result"]

    def submit_stream(self, cancel: threading.Event, **kwargs):
        """Enqueue a streaming job; returns a queue of ("delta"|"end"|...) events."""
        queue_out: queue.Queue = queue.Queue()
        self._jobs.put(
            {"kwargs": {**kwargs, "cancel": cancel}, "queue": queue_out}
        )
        return queue_out

    def _prepare(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> tuple[list[int], int]:
        from tinyllm.chat import build_prompt

        if not messages:
            raise ValueError("messages must not be empty")
        if messages[-1].get("role") != "user":
            raise ValueError("the last message must have role 'user'")

        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:-1]
            if m.get("role") in ("user", "assistant")
        ]
        question = messages[-1]["content"]
        if not isinstance(question, str):
            raise ValueError("message content must be a string")

        # Clients such as LocalLLM Studio ask for 16 000 tokens by default.
        # Cap generation at half the context so the prompt keeps room for the
        # question and history, and never produce a negative prompt budget.
        max_tokens = max(1, min(int(max_tokens), self._context_len // 2))
        budget = self._context_len - max_tokens
        prompt_ids = build_prompt(turns, question, self._tokenizer, budget)
        if not prompt_ids:
            raise ValueError("empty prompt after truncation")
        room = min(max_tokens, self._context_len - len(prompt_ids))
        if room <= 0:
            raise ValueError("prompt fills the whole context window")
        return prompt_ids, room

    def _stream(
        self,
        messages: list[dict],
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        max_tokens: int,
        seed: int,
        cancel: threading.Event,
    ):
        """Yield ("delta", text) events, then ("end", usage).

        MLX has no incremental API here, so the loop from
        ``tinyllm.sample.generate_ids`` is inlined to emit text as it appears.
        """
        import mlx.core as mx

        from tinyllm.sample import (
            apply_no_repeat_ngram_mask,
            apply_repetition_penalty,
            sample_token,
        )

        prompt_ids, room = self._prepare(messages, max_tokens)
        tokenizer = self._tokenizer

        cache = self._model.make_cache()
        prompt_started = time.perf_counter()
        logits = self._model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)[0, -1, :]
        mx.eval(logits)  # MLX is lazy: force the prompt pass out of the gen timing
        prompt_eval_ns = int((time.perf_counter() - prompt_started) * 1e9)
        key = mx.random.key(seed)

        generated: list[int] = []
        sent = ""
        gen_started = time.perf_counter()
        for _ in range(room):
            if cancel.is_set() or cache[0].length >= self._context_len:
                break

            logits = apply_repetition_penalty(
                logits, prompt_ids + generated, repetition_penalty
            )
            if no_repeat_ngram_size > 0:
                logits = apply_no_repeat_ngram_mask(
                    logits, prompt_ids + generated, n=no_repeat_ngram_size
                )

            key, subkey = mx.random.split(key)
            token = sample_token(logits, temperature, subkey, top_k, top_p)
            if token == tokenizer.eos_id or token == tokenizer.user_id:
                break

            generated.append(token)
            text = tokenizer.decode(generated)
            if len(text) > len(sent):
                yield ("delta", text[len(sent):])
                sent = text

            logits = self._model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1, :]

        yield (
            "end",
            {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "text": sent.strip(),
                "eval_duration": int((time.perf_counter() - gen_started) * 1e9),
                "prompt_eval_duration": prompt_eval_ns,
            },
        )

    def _complete(
        self,
        messages: list[dict],
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        max_tokens: int,
        seed: int,
    ) -> dict:
        from tinyllm.sample import generate_ids

        prompt_ids, room = self._prepare(messages, max_tokens)

        started = time.perf_counter()
        out = generate_ids(
            self._model,
            prompt_ids,
            max_tokens=room,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stop_at_eos=True,
            eos_id=self._tokenizer.eos_id,
            stop_on_user_token=True,
            user_id=self._tokenizer.user_id,
            seed=seed,
        )
        elapsed_ns = int((time.perf_counter() - started) * 1e9)
        prompt_tokens = len(prompt_ids)
        completion_tokens = len(out) - prompt_tokens
        return {
            "text": self._tokenizer.decode(out[prompt_tokens:]).strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "eval_duration": elapsed_ns,
            "prompt_eval_duration": 0,
        }


def load_bundle(model_dir: Path):
    from tinyllm.quantize import load_model

    metadata = json.loads((model_dir / "config.json").read_text())
    if "quantization" not in metadata:
        raise SystemExit(
            f"{model_dir}/config.json has no 'quantization' block; "
            "pass --run-dir instead."
        )
    model, tokenizer = load_model(model_dir)
    return model, tokenizer, metadata["model"], metadata["quantization"]


def load_run(run_dir: Path, checkpoint: str):
    from tinyllm.sample import load_run_model

    model, tokenizer, cfg = load_run_model(run_dir, checkpoint)
    return model, tokenizer, copy.deepcopy(vars(cfg.model)), None


def make_handler(worker: InferenceWorker):
    info = worker.describe()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except (ConnectionResetError, BrokenPipeError):
                # Client vanished mid-request; nothing useful to log.
                self.close_connection = True

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, kind="invalid_request_error"):
            print(f"  ! http {status}: {message}", flush=True)
            self._send(
                status,
                {
                    "error": {
                        "message": repr(message) if status >= 500 else message,
                        "type": kind,
                    }
                },
            )

        def _write_chunk(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _sse(self, payload: dict) -> None:
            self._write_chunk(
                b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
            )

        @staticmethod
        def _extra_metrics(usage: dict) -> dict:
            """Ollama-style metric fields, on top of the OpenAI ``usage`` block.

            LocalLLM Studio looks for ``eval_count``/``eval_duration`` (the
            names Ollama uses) to display tokens per second, and ignores a
            plain ``usage`` object.
            """
            eval_ns = usage.get("eval_duration") or 0
            eval_s = eval_ns / 1e9
            completion = usage["completion_tokens"]
            return {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": completion,
                "total_tokens": usage["prompt_tokens"] + completion,
                "eval_count": completion,
                "prompt_eval_count": usage["prompt_tokens"],
                "eval_duration": eval_ns,
                "prompt_eval_duration": usage.get("prompt_eval_duration") or 0,
                "tokens_per_second": round(completion / eval_s, 2) if eval_s > 0 else 0.0,
            }

        def _stream_completion(self, params: dict) -> None:
            """Server-sent events, chunked transfer encoding."""
            cancel = threading.Event()
            created = int(time.time())
            completion_id = f"chatcmpl-{int(created * 1000)}"

            def chunk(delta: dict, finish_reason=None) -> dict:
                return {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": info["name"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }

            events = worker.submit_stream(cancel, **params)
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            usage = None
            try:
                self._sse(chunk({"role": "assistant", "content": ""}))
                while True:
                    kind, value = events.get()
                    if kind == "delta":
                        self._sse(chunk({"content": value}))
                    elif kind == "end":
                        usage = value
                    elif kind == "error":
                        self._sse(
                            {"error": {"message": str(value), "type": "server_error"}}
                        )
                        break
                    else:  # "done" sentinel
                        break
                if usage is not None:
                    # One final chunk carrying both finish_reason and usage:
                    # clients that look for usage only on the chunk that closes
                    # the stream (LocalLLM Studio does) then see it.
                    metrics = self._extra_metrics(usage)
                    self._sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": info["name"],
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                            "usage": {
                                "prompt_tokens": usage["prompt_tokens"],
                                "completion_tokens": usage["completion_tokens"],
                                "total_tokens": usage["prompt_tokens"]
                                + usage["completion_tokens"],
                            },
                            "metrics": metrics,
                            **metrics,
                        }
                    )
                self._write_chunk(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("  ! client disconnected during stream", flush=True)
            finally:
                cancel.set()

        def _route(self) -> str:
            """Normalise the path so a base URL ending in /v1 still works.

            Clients differ: some append ``/v1/models`` to the base URL, others
            append only ``/models``.  Collapsing repeated ``/v1`` segments and
            duplicate slashes keeps both working instead of returning 404.
            """
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            while "//" in path:
                path = path.replace("//", "/")
            while path.startswith("/v1/v1"):
                path = path[3:]
            return path

        def do_GET(self) -> None:
            route = self._route()
            if route == "/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "model": info["name"],
                        "memory_bytes": current_memory_bytes(),
                    },
                )
            elif route in ("/v1/models", "/models"):
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": info["name"],
                                "object": "model",
                                "created": 0,
                                "owned_by": "local",
                                "parameters": info["parameters"],
                                "context_length": info["context_len"],
                                "memory_bytes": current_memory_bytes(),
                            }
                        ],
                    },
                )
            else:
                self._error(404, f"unknown path {self.path}")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            try:
                self._handle_post()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as exc:  # never drop the connection silently
                try:
                    self._error(500, f"unhandled error: {exc}", kind="server_error")
                except Exception:
                    self.close_connection = True

        def _handle_post(self) -> None:
            route = self._route()
            if route not in ("/v1/chat/completions", "/chat/completions"):
                self._error(404, f"unknown path {self.path}")
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                self._error(400, "missing or oversized request body")
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._error(400, f"invalid JSON: {exc}")
                return
            if not isinstance(payload, dict):
                self._error(400, "request body must be a JSON object")
                return

            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                self._error(400, "'messages' must be a non-empty list")
                return

            def number(key: str, default, cast=float):
                """Clients happily send null for sampling parameters."""
                value = payload.get(key, default)
                if value is None or value == "":
                    value = default
                return cast(value)

            params = dict(
                messages=messages,
                temperature=number("temperature", 0.5),
                top_k=number("top_k", 40, int),
                top_p=number("top_p", 0.9),
                repetition_penalty=number("repetition_penalty", 1.05),
                no_repeat_ngram_size=number("no_repeat_ngram_size", 4, int),
                max_tokens=number("max_tokens", 60, int),
                seed=number("seed", 42, int),
            )
            print(
                f"  -> chat: {len(messages)} messages, stream="
                f"{bool(payload.get('stream'))}, max_tokens={params['max_tokens']}, "
                f"temperature={params['temperature']}",
                flush=True,
            )
            extra = sorted(
                k
                for k in payload
                if k not in ("messages", "stream", "max_tokens", "temperature")
            )
            if extra:
                print(f"     payload keys: {extra}", flush=True)

            if payload.get("stream"):
                self._stream_completion(params)
                return

            started = time.time()
            try:
                result = worker.submit(**params)
            except (ValueError, TypeError) as exc:
                self._error(400, str(exc))
                return
            except Exception as exc:  # keep the server alive on generation bugs
                self._error(500, f"generation failed: {exc}", kind="server_error")
                return

            self._send(
                200,
                {
                    "id": f"chatcmpl-{int(started * 1000)}",
                    "object": "chat.completion",
                    "created": int(started),
                    "model": info["name"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result["text"]},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                    },
                    "metrics": self._extra_metrics(result),
                    **self._extra_metrics(result),
                    "elapsed_s": round(time.time() - started, 3),
                },
            )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path, help="quantized export bundle")
    source.add_argument("--run-dir", type=Path, help="training run directory")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--name", default="tinyllm-v9")
    args = parser.parse_args()

    if args.model_dir is not None:
        loader = lambda: load_bundle(args.model_dir)  # noqa: E731
    else:
        loader = lambda: load_run(args.run_dir, args.checkpoint)  # noqa: E731

    worker = InferenceWorker(loader, name=args.name)
    info = worker.describe()
    quantization = info["quantization"]
    quant_label = (
        f"int{quantization['bits']}/g{quantization['group_size']}"
        if quantization
        else "fp32"
    )
    print(
        f"TinyLLM server: {info['name']} | {info['parameters']:,} parameters "
        f"| {quant_label} | context {info['context_len']}",
        flush=True,
    )
    print(f"Listening on http://{args.host}:{args.port}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(worker))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
