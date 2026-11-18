#!/usr/bin/env python3
"""Stream the Stage B pre-training sources to resumable local JSONL files.

The script never downloads the full datasets in this repository test: use
``--dry-run`` to inspect the plan, or ``--local-input-dir`` to exercise the
same writer/resume path on local fixtures.

Each source is written to ``<output-dir>/<source>.jsonl`` with one record per
line::

    {"text": "...", "source": "fr_wikipedia", "lang": "fr",
     "kind": "text", "doc_index": 0}

A per-source ``<source>.state.json`` file records the number of documents and
characters already written.  Re-running the same command resumes after the
last complete line and never duplicates a document.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

from tinyllm.pretrain_v8 import SOURCES, SourceSpec, json_dump


def parse_sources(values: list[str] | None) -> list[SourceSpec]:
    if not values:
        return list(SOURCES.values())
    unknown = sorted(set(values) - set(SOURCES))
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(unknown)}")
    return [SOURCES[name] for name in values]


def iter_hf(spec: SourceSpec, skip: int) -> Iterator[dict]:
    from datasets import DownloadConfig, load_dataset

    dataset = load_dataset(
        spec.dataset,
        spec.config,
        split=spec.split,
        streaming=True,
        download_config=DownloadConfig(max_retries=5),
    )
    if skip:
        dataset = dataset.skip(skip)
    for row in dataset:
        value = row.get(spec.field)
        if isinstance(value, str) and value.strip():
            yield {"text": value}


def iter_local(spec: SourceSpec, input_dir: Path, skip: int) -> Iterator[dict]:
    path = input_dir / f"{spec.name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(spec.field, record.get("text"))
            if isinstance(value, str) and value.strip():
                yield {"text": value}


def repair_raw_file(path: Path) -> tuple[int, int]:
    """Drop a trailing partial JSON line and recompute docs/chars.

    Only the last incomplete line of *our own* output can be removed.  A
    complete corpus is never touched.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return 0, 0
    with path.open("rb+") as handle:
        data = handle.read()
        last_newline = data.rfind(b"\n")
        if last_newline + 1 != len(data):
            handle.seek(last_newline + 1)
            handle.truncate()
            data = data[: last_newline + 1]
    documents = 0
    characters = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            documents += 1
            characters += len(record.get("text", ""))
    return documents, characters


def load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_source(
    spec: SourceSpec,
    output_dir: Path,
    *,
    max_docs: int | None,
    char_scale: float,
    local_input_dir: Path | None,
) -> dict:
    raw_path = output_dir / f"{spec.name}.jsonl"
    state_path = output_dir / f"{spec.name}.state.json"
    state = load_state(state_path)
    if state is not None:
        documents_done, characters_done = repair_raw_file(raw_path)
        if documents_done < int(state.get("documents_done", 0)):
            print(
                f"[{spec.name}] state said {state['documents_done']} docs but "
                f"the file has {documents_done}; using the file.",
                flush=True,
            )
    else:
        documents_done = 0
        characters_done = 0
        if raw_path.exists():
            raise FileExistsError(
                f"{raw_path} exists without a state file; move it or choose "
                "another output directory."
            )

    if state is not None and state.get("complete"):
        return {
            **state,
            "documents_done": documents_done,
            "chars_written": characters_done,
            "skipped": True,
        }

    budget = int(spec.char_budget * char_scale)
    if max_docs is not None:
        remaining = max(0, max_docs - documents_done)
        if remaining == 0:
            return {
                "source": spec.name,
                "documents_done": documents_done,
                "chars_written": characters_done,
                "char_budget": budget,
                "complete": False,
                "max_docs_reached": True,
                "skipped": True,
            }
    else:
        remaining = None

    iterator = (
        iter_local(spec, local_input_dir, documents_done)
        if local_input_dir is not None
        else iter_hf(spec, documents_done)
    )

    documents = documents_done
    characters = characters_done
    written_this_run = 0
    started = time.time()
    complete = False

    with raw_path.open("a", encoding="utf-8") as handle:
        try:
            for record in iterator:
                text = record["text"]
                payload = {
                    "text": text,
                    "source": spec.name,
                    "lang": spec.lang,
                    "kind": spec.kind,
                    "doc_index": documents,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                documents += 1
                characters += len(text)
                written_this_run += 1

                if written_this_run % 1000 == 0:
                    json_dump(
                        state_path,
                        {
                            "source": spec.name,
                            "documents_done": documents,
                            "chars_written": characters,
                            "char_budget": budget,
                            "complete": False,
                        },
                    )
                    elapsed = max(1e-9, time.time() - started)
                    print(
                        f"[{spec.name}] {documents:,} docs / "
                        f"{characters / 1e6:.1f} M chars / "
                        f"{written_this_run / elapsed:.1f} docs/s",
                        flush=True,
                    )

                if remaining is not None and written_this_run >= remaining:
                    break
                if characters >= budget:
                    complete = True
                    break
        finally:
            handle.flush()
            os.fsync(handle.fileno())

    if remaining is None and characters >= budget:
        complete = True
    if remaining is not None and written_this_run >= remaining:
        complete = False

    result = {
        "source": spec.name,
        "dataset": spec.dataset,
        "config": spec.config,
        "field": spec.field,
        "lang": spec.lang,
        "kind": spec.kind,
        "documents_done": documents,
        "chars_written": characters,
        "char_budget": budget,
        "complete": complete,
        "written_this_run": written_this_run,
        "elapsed_s": round(time.time() - started, 3),
    }
    json_dump(state_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Stage B sources.")
    parser.add_argument("--output-dir", default="data/v8/pretrain/raw")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum total documents per source (for smoke tests).",
    )
    parser.add_argument(
        "--char-scale",
        type=float,
        default=1.0,
        help="Multiply every source character budget (default 1.0).",
    )
    parser.add_argument(
        "--local-input-dir",
        default=None,
        help="Read <source>.jsonl fixtures instead of Hugging Face.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without network access.",
    )
    args = parser.parse_args()

    if args.max_docs is not None and args.max_docs < 1:
        parser.error("--max-docs must be positive.")
    if args.char_scale <= 0:
        parser.error("--char-scale must be positive.")

    specs = parse_sources(args.sources)
    output_dir = Path(args.output_dir)
    local_input_dir = Path(args.local_input_dir) if args.local_input_dir else None

    plan = {
        "output_dir": str(output_dir),
        "local_input_dir": str(local_input_dir) if local_input_dir else None,
        "max_docs": args.max_docs,
        "char_scale": args.char_scale,
        "sources": [
            {
                "name": spec.name,
                "dataset": spec.dataset,
                "config": spec.config,
                "split": spec.split,
                "field": spec.field,
                "lang": spec.lang,
                "kind": spec.kind,
                "target_tokens": spec.target_tokens,
                "char_budget": int(spec.char_budget * args.char_scale),
            }
            for spec in specs
        ],
        "total_char_budget": sum(
            int(spec.char_budget * args.char_scale) for spec in specs
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        print(f"=== {spec.name} ===", flush=True)
        results.append(
            run_source(
                spec,
                output_dir,
                max_docs=args.max_docs,
                char_scale=args.char_scale,
                local_input_dir=local_input_dir,
            )
        )
    print(json.dumps({"sources": results}, indent=2, ensure_ascii=False))
    if any(not result.get("complete") and not result.get("max_docs_reached") for result in results):
        print(
            "Some sources stopped before their character budget; re-run the "
            "same command to resume.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
