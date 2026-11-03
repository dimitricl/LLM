
"""Prepare contextual conversation windows for TinyLLM SFT.

The v1 packer cut a fully encoded conversation into independent windows of
``context_len`` tokens.  A window could therefore start in the middle of an
assistant answer, without the user question or the ASSISTANT marker.  Long
answers were learned as free-form continuation instead of conversational
replies.

This module builds *contextual* windows instead:

* every supervised assistant token belongs to exactly one target chunk;
* each chunk is preceded by a context prefix that contains, by priority,
  the current user question, a minimum amount of the preceding answer and
  recent conversation history;
* the first stored token of a row is always context, so the trainer mask
  ``mask[:, 1:]`` never drops a target because of the causal offset;
* padding is PAD with a zero mask.

Rows are still ``context_len + 1`` tokens with one mask value per token, so
``SFTSequenceDataset`` and the masked loss remain compatible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyllm.tokenizer import TokenizerWrapper

SEQUENCE_LAYOUT = "contextual_conversation_windows"
SUPPORTED_SEQUENCE_LAYOUTS = (
    "independent_conversation_windows",
    SEQUENCE_LAYOUT,
)
FORMAT_VERSION = 2

DEFAULT_MAX_QUESTION_TOKENS = 256
DEFAULT_MIN_RESPONSE_CONTEXT = 64
DEFAULT_MIN_HISTORY_CONTEXT = 32


# ---------------------------------------------------------------------------
# Conversation normalisation / splitting
# ---------------------------------------------------------------------------


def canonical_messages(messages: list[dict]) -> str:
    return json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_split(messages: list[dict], val_frac: float, test_frac: float) -> str:
    digest = hashlib.sha256(
        canonical_messages(messages).encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / (2**64)

    if value < test_frac:
        return "test"
    if value < test_frac + val_frac:
        return "val"
    return "train"


def normalize_messages(raw_messages: object) -> list[dict]:
    if not isinstance(raw_messages, list):
        raise ValueError("'messages' must be a list.")

    messages: list[dict] = []
    system_messages: list[str] = []

    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index} must be an object.")

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError(f"Message {index} content must be a string.")

        content = content.strip()
        if not content:
            continue

        if role == "system":
            if messages:
                raise ValueError("System messages must precede user messages.")
            system_messages.append(content)
            continue

        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported role: {role!r}")

        if messages and messages[-1]["role"] == role:
            raise ValueError(f"Consecutive messages with the same role: {role!r}")

        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("Conversation is empty.")

    if messages[0]["role"] != "user":
        raise ValueError("Conversation must begin with a user message.")

    if messages[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with an assistant message.")

    if system_messages:
        messages[0]["content"] = (
            "System instructions: "
            + "\n\n".join(system_messages)
            + "\n\n"
            + messages[0]["content"]
        )

    return messages


# ---------------------------------------------------------------------------
# Encoding with structural spans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantRegion:
    """A supervised assistant turn ``[marker, content, eos)``."""

    marker: int
    content_start: int
    content_end: int
    end: int
    user_start: int


@dataclass(frozen=True)
class EncodedConversation:
    ids: list[int]
    mask: list[int]
    regions: list[AssistantRegion]


def encode_conversation(
    messages: list[dict],
    tokenizer: TokenizerWrapper,
) -> EncodedConversation:
    ids: list[int] = []
    mask: list[int] = []
    regions: list[AssistantRegion] = []
    pending_user_start: int | None = None

    for message in messages:
        role = message["role"]
        content_ids = tokenizer.encode(message["content"])

        if role == "user":
            pending_user_start = len(ids)
            ids.append(tokenizer.user_id)
            mask.append(0)

            ids.extend(content_ids)
            mask.extend([0] * len(content_ids))
            continue

        if pending_user_start is None:
            raise ValueError("Assistant message has no preceding user message.")

        marker = len(ids)
        ids.append(tokenizer.assistant_id)
        mask.append(1)

        content_start = len(ids)
        ids.extend(content_ids)
        mask.extend([1] * len(content_ids))
        content_end = len(ids)

        ids.append(tokenizer.eos_id)
        mask.append(1)
        end = len(ids)

        regions.append(
            AssistantRegion(
                marker=marker,
                content_start=content_start,
                content_end=content_end,
                end=end,
                user_start=pending_user_start,
            )
        )
        pending_user_start = None

    if len(ids) < 2:
        raise ValueError("Conversation contains too few tokens.")

    if not regions:
        raise ValueError("Conversation contains no assistant response.")

    if max(ids) >= tokenizer.vocab_size:
        raise ValueError("Encoded token ID exceeds tokenizer vocabulary.")

    return EncodedConversation(ids=ids, mask=mask, regions=regions)


def encode_messages(
    messages: list[dict],
    tokenizer: TokenizerWrapper,
) -> tuple[list[int], list[int]]:
    """Backwards-compatible helper returning the flat ids and mask."""
    encoded = encode_conversation(messages, tokenizer)
    return encoded.ids, encoded.mask


# ---------------------------------------------------------------------------
# Contextual window construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowPolicy:
    context_len: int = 512
    max_question_tokens: int = DEFAULT_MAX_QUESTION_TOKENS
    min_response_context: int = DEFAULT_MIN_RESPONSE_CONTEXT
    min_history_context: int = DEFAULT_MIN_HISTORY_CONTEXT

    def __post_init__(self) -> None:
        if self.context_len < 1:
            raise ValueError("context_len must be positive.")
        if self.max_question_tokens < 2:
            raise ValueError("max_question_tokens must be at least 2.")
        if self.min_response_context < 0:
            raise ValueError("min_response_context cannot be negative.")
        if self.min_history_context < 0:
            raise ValueError("min_history_context cannot be negative.")
        if (
            self.max_question_tokens
            + 1
            + self.min_response_context
            >= self.context_len + 1
        ):
            raise ValueError(
                "Question and response reserves leave no room for targets."
            )
        if (
            self.max_question_tokens + self.min_history_context
            >= self.context_len + 1
        ):
            raise ValueError(
                "Question and history reserves leave no room for targets."
            )

    @property
    def sequence_len(self) -> int:
        return self.context_len + 1


@dataclass(frozen=True)
class Window:
    tokens: list[int]
    mask: list[int]
    kind: str
    target_start: int
    target_end: int
    prefix_len: int


def _initial_prefix(
    history_ids: list[int],
    question_ids: list[int],
    budget: int,
) -> list[int]:
    """Question first, then as much recent history as the budget allows."""
    if len(question_ids) > budget:
        raise ValueError("Question does not fit the context budget.")

    room = budget - len(question_ids)
    history_len = min(len(history_ids), room)
    prefix = history_ids[-history_len:] if history_len else []
    return prefix + question_ids


def _continuation_prefix(
    history_ids: list[int],
    question_ids: list[int],
    conversation_ids: list[int],
    region: AssistantRegion,
    target_start: int,
    budget: int,
    assistant_id: int,
) -> list[int]:
    """Question + ASSISTANT marker + tail of the answer written so far."""
    if len(question_ids) + 1 > budget:
        raise ValueError("Question and marker do not fit the context budget.")

    # Never re-emit the original ASSISTANT marker: the marker in the prefix is
    # synthetic, and the preceding answer text is content only.
    prior_content = target_start - region.content_start
    room = budget - len(question_ids) - 1

    response_len = min(prior_content, room)
    room -= response_len

    history_len = min(len(history_ids), room)
    history_prefix = history_ids[-history_len:] if history_len else []

    response_prefix = (
        conversation_ids[target_start - response_len : target_start]
        if response_len
        else []
    )

    return history_prefix + question_ids + [assistant_id] + response_prefix


def build_conversation_windows(
    encoded: EncodedConversation,
    tokenizer: TokenizerWrapper,
    policy: WindowPolicy,
) -> tuple[list[Window], int]:
    """Split every assistant turn into context-carrying target chunks.

    Returns the windows and the number of truncated user questions.

    Target coverage is a partition of each assistant region: chunks are
    contiguous, non-overlapping, and cover ``[marker, eos]`` exactly once.
    """
    sequence_len = policy.sequence_len
    windows: list[Window] = []
    truncated_questions = 0

    for region in encoded.regions:
        question_full = encoded.ids[region.user_start : region.marker]

        if len(question_full) > policy.max_question_tokens:
            truncated_questions += 1
            question = question_full[: policy.max_question_tokens]
        else:
            question = question_full

        history_ids = encoded.ids[: region.user_start]
        target_start = region.marker
        first_chunk = True

        while target_start < region.end:
            if first_chunk:
                # The marker is the first target and must be supervised.
                required_prefix = len(question) + min(
                    len(history_ids), policy.min_history_context
                )
                kind = "initial"
            else:
                prior_content = target_start - region.content_start
                required_prefix = (
                    len(question)
                    + 1
                    + min(prior_content, policy.min_response_context)
                )
                kind = "continuation"

            if required_prefix >= sequence_len:
                raise ValueError(
                    "Context reserves leave no room for a target token."
                )

            max_targets = min(
                sequence_len - required_prefix,
                policy.context_len,
            )
            remaining = region.end - target_start
            target_len = min(remaining, max_targets)

            if target_len < 1:
                raise ValueError("Window target chunk is empty.")

            # Do not leave a single EOS as its own window when the previous
            # chunk can absorb the split point differently.
            if remaining - target_len == 1 and target_len > 1:
                target_len -= 1

            target_end = target_start + target_len
            budget = sequence_len - target_len

            if first_chunk:
                prefix = _initial_prefix(history_ids, question, budget)
            else:
                prefix = _continuation_prefix(
                    history_ids,
                    question,
                    encoded.ids,
                    region,
                    target_start,
                    budget,
                    tokenizer.assistant_id,
                )

            if len(prefix) < 1:
                raise ValueError("A window must start with context.")
            if len(prefix) + target_len > sequence_len:
                raise ValueError("Window exceeds the sequence length.")

            tokens = prefix + encoded.ids[target_start:target_end]
            mask = [0] * len(prefix) + [1] * target_len

            padding = sequence_len - len(tokens)
            tokens = tokens + [tokenizer.pad_id] * padding
            mask = mask + [0] * padding

            windows.append(
                Window(
                    tokens=tokens,
                    mask=mask,
                    kind=kind,
                    target_start=target_start,
                    target_end=target_end,
                    prefix_len=len(prefix),
                )
            )

            target_start = target_end
            first_chunk = False

        if target_start != region.end:
            raise ValueError("Assistant region was not fully covered.")

    return windows, truncated_questions


# ---------------------------------------------------------------------------
# Binary output / statistics
# ---------------------------------------------------------------------------


SPLIT_NAMES = ("train", "val", "test")


def _empty_split_stats() -> dict:
    return {
        "conversations": 0,
        "conversations_rejected": 0,
        "sequences": 0,
        "initial_windows": 0,
        "continuation_windows": 0,
        "responses": 0,
        "long_responses": 0,
        "questions_truncated": 0,
        "multi_turn_conversations": 0,
        # Filled from the binary files during finalisation.
        "tokens": 0,
        "context_tokens": 0,
        "target_tokens": 0,
        "assistant_tokens": 0,
        "padding_tokens": 0,
    }


def _empty_source_stats() -> dict:
    return {
        "examined": 0,
        "accepted": 0,
        "duplicates": 0,
        "rejected": 0,
        "rejected_reasons": Counter(),
    }


def _write_window(token_handle, mask_handle, window: Window) -> None:
    if len(window.tokens) != len(window.mask):
        raise ValueError("Window token/mask length mismatch.")

    np.asarray(window.tokens, dtype=np.uint16).tofile(token_handle)
    np.asarray(window.mask, dtype=np.uint8).tofile(mask_handle)


def _scan_split_binary(
    token_path: Path,
    mask_path: Path,
    sequence_len: int,
    pad_id: int,
) -> dict:
    """Recompute statistics from the files actually written.

    The scan also validates the row layout: a non-padding prefix, then a
    contiguous target suffix, then padding only.
    """
    token_bytes = token_path.stat().st_size
    mask_bytes = mask_path.stat().st_size
    token_row_bytes = sequence_len * np.dtype(np.uint16).itemsize
    mask_row_bytes = sequence_len * np.dtype(np.uint8).itemsize

    if token_bytes == 0 or mask_bytes == 0:
        return {
            "sequences": 0,
            "tokens": 0,
            "context_tokens": 0,
            "target_tokens": 0,
            "assistant_tokens": 0,
            "padding_tokens": 0,
        }

    if token_bytes % token_row_bytes != 0 or mask_bytes % mask_row_bytes != 0:
        raise ValueError("SFT binary size is not aligned to the sequence length.")

    token_rows = token_bytes // token_row_bytes
    mask_rows = mask_bytes // mask_row_bytes

    if token_rows != mask_rows:
        raise ValueError("Token/mask row count mismatch.")

    tokens = np.memmap(
        token_path, dtype=np.uint16, mode="r", shape=(token_rows, sequence_len)
    )
    masks = np.memmap(
        mask_path, dtype=np.uint8, mode="r", shape=(mask_rows, sequence_len)
    )

    context_tokens = 0
    target_tokens = 0
    assistant_tokens = 0
    padding_tokens = 0

    chunk_rows = 4096

    for start in range(0, token_rows, chunk_rows):
        stop = min(start + chunk_rows, token_rows)
        token_chunk = np.asarray(tokens[start:stop])
        mask_chunk = np.asarray(masks[start:stop])

        if np.any(mask_chunk > 1):
            raise ValueError("Mask contains values other than 0/1.")

        if np.any(mask_chunk[:, 0] != 0):
            raise ValueError(
                "First mask value must be zero: the causal loss drops it."
            )

        # The last real token is the last supervised target; everything after
        # it is padding.  Inferring the boundary from the mask (instead of the
        # PAD id) keeps the check valid even if a content token happens to use
        # the PAD id.
        reversed_mask = mask_chunk[:, ::-1]
        if not np.all(reversed_mask.max(axis=1) == 1):
            raise ValueError("A sequence contains no supervised target.")

        last_target = mask_chunk.shape[1] - 1 - np.argmax(reversed_mask, axis=1)
        real_len = last_target + 1
        target_per_row = mask_chunk.sum(axis=1).astype(np.int64)
        context_per_row = real_len - target_per_row

        if np.any(context_per_row < 1):
            raise ValueError("A sequence has no context token before its targets.")

        columns = np.arange(mask_chunk.shape[1])
        before_real = columns[None, :] < real_len[:, None]
        after_real = ~before_real

        # Targets must be a contiguous suffix of the real tokens: a 0 after a
        # 1 (before padding) means a target would be lost or misaligned.
        seen_target = np.maximum.accumulate(mask_chunk, axis=1)
        if np.any((seen_target == 1) & (mask_chunk == 0) & before_real):
            raise ValueError("Supervised targets are not contiguous.")

        if np.any(token_chunk[after_real] != pad_id):
            raise ValueError("Padding positions do not contain the PAD token.")

        target_tokens += int(target_per_row.sum())
        assistant_tokens += int(mask_chunk[:, 1:].sum())
        context_tokens += int(context_per_row.sum())
        padding_tokens += int((mask_chunk.shape[1] - real_len).sum())

    return {
        "sequences": int(token_rows),
        "tokens": int(token_rows) * sequence_len,
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "assistant_tokens": assistant_tokens,
        "padding_tokens": padding_tokens,
    }


def _fsync_directory(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


# ---------------------------------------------------------------------------
# Packer
# ---------------------------------------------------------------------------


def prepare(
    input_paths: list[Path],
    tokenizer_path: Path,
    output_dir: Path,
    context_len: int = 512,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    min_assistant_chars: int = 1,
    max_question_tokens: int = DEFAULT_MAX_QUESTION_TOKENS,
    min_response_context: int = DEFAULT_MIN_RESPONSE_CONTEXT,
    min_history_context: int = DEFAULT_MIN_HISTORY_CONTEXT,
) -> dict:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 0.5:
        raise ValueError(
            "Validation and test fractions must be >= 0 and sum below 0.5."
        )

    if min_assistant_chars < 0:
        raise ValueError("min_assistant_chars cannot be negative.")

    policy = WindowPolicy(
        context_len=context_len,
        max_question_tokens=max_question_tokens,
        min_response_context=min_response_context,
        min_history_context=min_history_context,
    )

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Choose a new output directory "
            "or remove it explicitly after checking its contents."
        )

    tokenizer = TokenizerWrapper(tokenizer_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )

    handles: dict[str, tuple] = {}
    split_stats = {name: _empty_split_stats() for name in SPLIT_NAMES}
    source_stats: dict[str, dict] = {}
    rejected_reasons: Counter = Counter()
    rejected_before_split = 0
    rejected_no_windows = 0
    seen_hashes: set[str] = set()

    def close_handles() -> None:
        for pair in handles.values():
            for handle in pair:
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
                try:
                    handle.close()
                except Exception:
                    pass

    try:
        for split in SPLIT_NAMES:
            handles[split] = (
                (staging_dir / f"{split}.tokens.bin").open("wb"),
                (staging_dir / f"{split}.mask.bin").open("wb"),
            )

        for input_path in input_paths:
            source_name = input_path.name
            source = source_stats.setdefault(source_name, _empty_source_stats())

            with input_path.open("r", encoding="utf-8") as source_file:
                for line in source_file:
                    if not line.strip():
                        continue

                    source["examined"] += 1

                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("JSONL record must be an object.")

                        messages = normalize_messages(record.get("messages"))

                        assistant_chars = sum(
                            len(message["content"])
                            for message in messages
                            if message["role"] == "assistant"
                        )

                        if assistant_chars < min_assistant_chars:
                            raise ValueError("Assistant answer is too short.")

                        canonical = canonical_messages(messages)
                        digest = hashlib.sha256(
                            canonical.encode("utf-8")
                        ).hexdigest()

                        if digest in seen_hashes:
                            source["duplicates"] += 1
                            continue

                        encoded = encode_conversation(messages, tokenizer)

                    except Exception as exc:
                        source["rejected"] += 1
                        rejected_before_split += 1
                        reason = str(exc)
                        source["rejected_reasons"][reason] += 1
                        rejected_reasons[reason] += 1
                        continue

                    seen_hashes.add(digest)
                    split = stable_split(messages, val_frac, test_frac)
                    windows, truncated_questions = build_conversation_windows(
                        encoded, tokenizer, policy
                    )

                    if not windows:
                        split_stats[split]["conversations_rejected"] += 1
                        source["rejected"] += 1
                        rejected_no_windows += 1
                        reason = "no_supervised_sequence"
                        source["rejected_reasons"][reason] += 1
                        rejected_reasons[reason] += 1
                        continue

                    token_handle, mask_handle = handles[split]

                    for window in windows:
                        _write_window(token_handle, mask_handle, window)

                    stats = split_stats[split]
                    initial = sum(1 for window in windows if window.kind == "initial")
                    continuation = len(windows) - initial

                    stats["conversations"] += 1
                    stats["sequences"] += len(windows)
                    stats["initial_windows"] += initial
                    stats["continuation_windows"] += continuation
                    stats["responses"] += len(encoded.regions)
                    stats["long_responses"] += sum(
                        1
                        for region in encoded.regions
                        if region.content_end - region.content_start > context_len
                    )
                    stats["questions_truncated"] += truncated_questions
                    stats["multi_turn_conversations"] += int(
                        len(encoded.regions) > 1
                    )

                    source["accepted"] += 1

        close_handles()
        handles = {}

        for split in SPLIT_NAMES:
            expected_empty = (
                split == "train"
                or (split == "val" and val_frac > 0)
                or (split == "test" and test_frac > 0)
            )
            if split_stats[split]["sequences"] == 0 and expected_empty:
                raise ValueError(
                    f"No usable sequences were produced for {split}."
                )

        for split in SPLIT_NAMES:
            actual = _scan_split_binary(
                staging_dir / f"{split}.tokens.bin",
                staging_dir / f"{split}.mask.bin",
                policy.sequence_len,
                tokenizer.pad_id,
            )
            stats = split_stats[split]

            if actual["sequences"] != stats["sequences"]:
                raise RuntimeError(
                    f"{split}: sequence count mismatch between write and disk."
                )

            if actual["assistant_tokens"] != actual["target_tokens"]:
                raise RuntimeError(
                    f"{split}: target mask is not aligned with the loss mask."
                )

            if stats["initial_windows"] + stats["continuation_windows"] != actual[
                "sequences"
            ]:
                raise RuntimeError(
                    f"{split}: initial + continuation != sequences."
                )

            stats.update(actual)

        accepted_total = sum(
            stats["conversations"] for stats in split_stats.values()
        )
        source_accepted = sum(source["accepted"] for source in source_stats.values())

        if accepted_total != source_accepted:
            raise RuntimeError(
                "Split conversation total differs from accepted source total."
            )

        for split in SPLIT_NAMES:
            stats = split_stats[split]
            if (
                stats["initial_windows"] + stats["continuation_windows"]
                != stats["sequences"]
            ):
                raise RuntimeError(
                    f"{split}: window kind totals do not add up."
                )

        metadata = {
            "format_version": FORMAT_VERSION,
            "sequence_layout": SEQUENCE_LAYOUT,
            "token_dtype": "uint16",
            "mask_dtype": "uint8",
            "mask_alignment": "one mask value per token; training uses mask[1:]",
            "context_policy": {
                "sequence_len": policy.sequence_len,
                "context_len": policy.context_len,
                "max_question_tokens": policy.max_question_tokens,
                "min_response_context": policy.min_response_context,
                "min_history_context": policy.min_history_context,
                "question_truncation": (
                    "keep the first max_question_tokens tokens of the current "
                    "user turn (head truncation)"
                ),
                "priority": [
                    "current user question",
                    "assistant marker",
                    "recent answer context (continuations)",
                    "recent conversation history (initial windows)",
                    "new supervised targets",
                ],
                "target_coverage": (
                    "each assistant region is partitioned into contiguous, "
                    "non-overlapping target chunks; answer context reused by a "
                    "continuation is stored with mask 0"
                ),
                "padding": "PAD tokens after the target chunk, mask 0",
            },
            "tokenizer": str(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "eos_id": tokenizer.eos_id,
            "pad_id": tokenizer.pad_id,
            "user_id": tokenizer.user_id,
            "assistant_id": tokenizer.assistant_id,
            "context_len": context_len,
            "sequence_len": policy.sequence_len,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "long_response_threshold": context_len,
            "sources": [str(path) for path in input_paths],
            "source_stats": {
                name: {
                    **stats,
                    "rejected_reasons": dict(stats["rejected_reasons"]),
                }
                for name, stats in source_stats.items()
            },
            "rejected_reasons": dict(rejected_reasons),
            "rejected_before_split": rejected_before_split,
            "rejected_no_windows": rejected_no_windows,
            "splits": split_stats,
            "totals": {
                "conversations": accepted_total,
                "sequences": sum(s["sequences"] for s in split_stats.values()),
                "assistant_tokens": sum(
                    s["assistant_tokens"] for s in split_stats.values()
                ),
                "initial_windows": sum(
                    s["initial_windows"] for s in split_stats.values()
                ),
                "continuation_windows": sum(
                    s["continuation_windows"] for s in split_stats.values()
                ),
            },
            "invariants": {
                "total_sequences_is_initial_plus_continuation": all(
                    stats["initial_windows"] + stats["continuation_windows"]
                    == stats["sequences"]
                    for stats in split_stats.values()
                ),
                "supervised_tokens_equal_mask_sum": all(
                    stats["assistant_tokens"] == stats["target_tokens"]
                    for stats in split_stats.values()
                ),
                "split_conversations_equal_accepted_sources": (
                    accepted_total == source_accepted
                ),
                "each_conversation_in_one_split": True,
            },
        }

        meta_path = staging_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(staging_dir, output_dir)
        try:
            _fsync_directory(output_dir.parent)
        except OSError:
            # Durability of the rename is best effort; the dataset itself is
            # already complete and atomically visible.
            pass

        return metadata

    except BaseException:
        close_handles()
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare contextual conversation windows for TinyLLM SFT."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more JSONL conversation files.",
    )
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--output-dir", default="data/sft/packed_v2")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--test-frac", type=float, default=0.02)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument(
        "--max-question-tokens",
        type=int,
        default=DEFAULT_MAX_QUESTION_TOKENS,
    )
    parser.add_argument(
        "--min-response-context",
        type=int,
        default=DEFAULT_MIN_RESPONSE_CONTEXT,
    )
    parser.add_argument(
        "--min-history-context",
        type=int,
        default=DEFAULT_MIN_HISTORY_CONTEXT,
    )

    args = parser.parse_args()

    result = prepare(
        input_paths=[Path(path) for path in args.input],
        tokenizer_path=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        context_len=args.context_len,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_assistant_chars=args.min_assistant_chars,
        max_question_tokens=args.max_question_tokens,
        min_response_context=args.min_response_context,
        min_history_context=args.min_history_context,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
