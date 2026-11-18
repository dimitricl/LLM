import json
from pathlib import Path

import numpy as np
import pytest

from tinyllm.sft_data import (
    DEFAULT_MAX_QUESTION_TOKENS,
    WindowPolicy,
    build_conversation_windows,
    encode_conversation,
    normalize_messages,
    prepare,
    stable_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def conversation(*turns, system: str | None = None) -> dict:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend({"role": role, "content": content} for role, content in turns)
    return {"messages": messages}


def encoded_for(record: dict, tokenizer):
    messages = normalize_messages(record["messages"])
    return messages, encode_conversation(messages, tokenizer)


def window_pair(record: dict, tokenizer, policy: WindowPolicy | None = None):
    _, encoded = encoded_for(record, tokenizer)
    return encoded, build_conversation_windows(
        encoded, tokenizer, policy or WindowPolicy()
    )


def assert_window_layout(window, tokenizer, sequence_len: int = 513) -> None:
    assert len(window.tokens) == sequence_len
    assert len(window.mask) == sequence_len
    assert window.mask[0] == 0, "first mask value is dropped by mask[:, 1:]"
    assert window.prefix_len >= 1, "a window needs context before its targets"

    target_len = window.target_end - window.target_start
    assert target_len >= 1
    assert window.prefix_len + target_len <= sequence_len

    prefix_mask = window.mask[: window.prefix_len]
    target_mask = window.mask[window.prefix_len : window.prefix_len + target_len]
    padding_mask = window.mask[window.prefix_len + target_len :]

    assert all(value == 0 for value in prefix_mask)
    assert all(value == 1 for value in target_mask)
    assert all(value == 0 for value in padding_mask)

    padding_tokens = window.tokens[window.prefix_len + target_len :]
    assert all(value == tokenizer.pad_id for value in padding_tokens)


def assert_full_coverage(encoded, windows) -> None:
    expected = []
    for region in encoded.regions:
        expected.extend(range(region.marker, region.end))

    covered = []
    for window in windows:
        covered.extend(range(window.target_start, window.target_end))

    assert covered == expected
    assert len(covered) == len(set(covered))


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def read_split(output_dir: Path, split: str, sequence_len: int = 513):
    tokens = np.fromfile(output_dir / f"{split}.tokens.bin", dtype=np.uint16)
    masks = np.fromfile(output_dir / f"{split}.mask.bin", dtype=np.uint8)
    assert tokens.size % sequence_len == 0
    assert masks.size % sequence_len == 0
    return tokens.reshape(-1, sequence_len), masks.reshape(-1, sequence_len)


# ---------------------------------------------------------------------------
# Encoding and context
# ---------------------------------------------------------------------------


def test_simple_conversation_single_window(tiny_tokenizer):
    record = conversation(
        ("user", "Bonjour, comment vas-tu ?"),
        ("assistant", "Je vais bien, merci."),
    )

    encoded, (windows, truncated) = window_pair(record, tiny_tokenizer)

    assert len(encoded.regions) == 1
    assert truncated == 0
    assert len(windows) == 1

    window = windows[0]
    assert window.kind == "initial"
    assert_window_layout(window, tiny_tokenizer)
    assert window.target_start == encoded.regions[0].marker
    assert window.target_end == encoded.regions[0].end

    target_len = window.target_end - window.target_start
    targets = window.tokens[window.prefix_len : window.prefix_len + target_len]

    assert targets[0] == tiny_tokenizer.assistant_id
    assert window.mask[window.prefix_len] == 1
    assert targets[-1] == tiny_tokenizer.eos_id
    assert window.mask[window.prefix_len + target_len - 1] == 1


def test_multi_turn_uses_previous_exchange_as_context(tiny_tokenizer):
    record = conversation(
        ("user", "Quelle est la capitale de la France ?"),
        ("assistant", "Paris est la capitale de la France."),
        ("user", "Et combien d'habitants compte-t-elle ?"),
        ("assistant", "Paris compte environ deux millions d'habitants."),
    )

    encoded, (windows, truncated) = window_pair(record, tiny_tokenizer)

    assert len(encoded.regions) == 2
    assert truncated == 0
    assert len(windows) == 2
    assert all(window.kind == "initial" for window in windows)

    second_prefix = tiny_tokenizer.decode(
        windows[1].tokens[: windows[1].prefix_len]
    )
    assert "capitale de la France" in second_prefix
    assert "combien d'habitants" in second_prefix

    for window in windows:
        assert_window_layout(window, tiny_tokenizer)

    assert_full_coverage(encoded, windows)


def test_short_response_stays_in_one_window(tiny_tokenizer):
    record = conversation(
        ("user", "Dis oui."),
        ("assistant", "Oui."),
    )
    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    assert len(windows) == 1
    assert windows[0].kind == "initial"
    assert_full_coverage(encoded, windows)


def test_long_response_is_split_with_continuation_context(tiny_tokenizer):
    answer = " ".join(f"mot{n}" for n in range(1400))
    record = conversation(
        ("user", "Explique longuement le sujet."),
        ("assistant", answer),
    )

    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    assert len(windows) > 1
    assert windows[0].kind == "initial"
    assert all(window.kind == "continuation" for window in windows[1:])

    for window in windows:
        assert_window_layout(window, tiny_tokenizer)

    assert_full_coverage(encoded, windows)

    for previous, current in zip(windows, windows[1:], strict=False):
        assert current.target_start == previous.target_end
        assert current.mask[current.prefix_len] == 1

    for window in windows[1:]:
        prefix = tiny_tokenizer.decode(window.tokens[: window.prefix_len])
        assert "Explique longuement" in prefix
        assert tiny_tokenizer.assistant_id in window.tokens[: window.prefix_len]


def test_very_long_response_has_no_lost_or_duplicated_targets(tiny_tokenizer):
    answer = " ".join(f"token{n}" for n in range(6000))
    record = conversation(
        ("user", "Donne une réponse très longue."),
        ("assistant", answer),
    )

    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    assert len(windows) > 5
    assert_full_coverage(encoded, windows)

    region = encoded.regions[0]
    assert windows[-1].target_end == region.end

    last = windows[-1]
    last_target_index = last.prefix_len + (last.target_end - last.target_start) - 1
    assert last.tokens[last_target_index] == tiny_tokenizer.eos_id

    for window in windows:
        assert_window_layout(window, tiny_tokenizer)


def test_long_question_is_truncated_deterministically(tiny_tokenizer):
    question = " ".join(f"question{n}" for n in range(400))
    policy = WindowPolicy(max_question_tokens=24)
    record = conversation(
        ("user", question),
        ("assistant", "Réponse courte."),
    )

    encoded, (windows, truncated) = window_pair(record, tiny_tokenizer, policy)

    assert truncated == 1
    assert len(windows) == 1

    region = encoded.regions[0]
    question_ids = encoded.ids[region.user_start : region.marker]
    expected_prefix = question_ids[:24]

    assert windows[0].tokens[: windows[0].prefix_len] == expected_prefix
    assert windows[0].prefix_len == 24


def test_default_question_budget_matches_policy(tiny_tokenizer):
    question = " ".join(f"question{n}" for n in range(1000))
    record = conversation(
        ("user", question),
        ("assistant", "Réponse courte."),
    )

    encoded, (windows, truncated) = window_pair(record, tiny_tokenizer)

    assert truncated == 1
    region = encoded.regions[0]
    question_ids = encoded.ids[region.user_start : region.marker]
    assert len(question_ids) > DEFAULT_MAX_QUESTION_TOKENS
    assert windows[0].prefix_len == DEFAULT_MAX_QUESTION_TOKENS


def test_short_question_long_answer_keeps_question_in_every_window(tiny_tokenizer):
    answer = " ".join(f"détail{n}" for n in range(1200))
    record = conversation(
        ("user", "Quelle est la recette ?"),
        ("assistant", answer),
    )

    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    assert len(windows) > 1
    region = encoded.regions[0]

    for window in windows:
        prefix = tiny_tokenizer.decode(window.tokens[: window.prefix_len])
        assert "Quelle est la recette" in prefix

    for window in windows[1:]:
        # The last context token is the original token immediately before the
        # first new target: the answer tail is contiguous with the targets.
        assert window.tokens[window.prefix_len - 1] == encoded.ids[
            window.target_start - 1
        ]
        assert (
            window.target_start - region.content_start
            >= WindowPolicy().min_response_context
        )


def test_code_response_keeps_mask_alignment(tiny_tokenizer):
    code = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "for value in range(10):\n"
        "    print(add(value, 1))\n"
    ) * 80
    record = conversation(
        ("user", "Écris du code Python."),
        ("assistant", code),
    )

    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    for window in windows:
        assert_window_layout(window, tiny_tokenizer)

    assert_full_coverage(encoded, windows)

    first_target = windows[0].tokens[
        windows[0].prefix_len : windows[0].prefix_len
        + (windows[0].target_end - windows[0].target_start)
    ]
    assert tiny_tokenizer.assistant_id in first_target


def test_system_instructions_are_preserved_in_first_question(tiny_tokenizer):
    record = conversation(
        ("user", "Résume ce texte."),
        ("assistant", "Voici un résumé."),
        system="Tu es un assistant concis et tu réponds en français.",
    )

    messages, encoded = encoded_for(record, tiny_tokenizer)
    windows, _ = build_conversation_windows(
        encoded, tiny_tokenizer, WindowPolicy()
    )

    assert messages[0]["content"].startswith("System instructions:")
    prefix = tiny_tokenizer.decode(windows[0].tokens[: windows[0].prefix_len])
    assert "System instructions:" in prefix
    assert "concis" in prefix


# ---------------------------------------------------------------------------
# Mask semantics
# ---------------------------------------------------------------------------


def test_mask_alignment_and_no_target_at_row_start(tiny_tokenizer):
    answer = " ".join(f"item{n}" for n in range(900))
    record = conversation(
        ("user", "Question simple."),
        ("assistant", answer),
    )

    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    for window in windows:
        target_len = window.target_end - window.target_start
        assert window.mask[0] == 0
        assert window.mask[window.prefix_len - 1] == 0
        assert sum(window.mask) == target_len
        assert sum(window.mask[1:]) == target_len


def test_assistant_marker_is_supervised_in_initial_window(tiny_tokenizer):
    record = conversation(
        ("user", "Bonjour."),
        ("assistant", "Bonjour à toi."),
    )
    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    region = encoded.regions[0]
    window = windows[0]
    marker_index = window.prefix_len

    assert window.target_start == region.marker
    assert window.tokens[marker_index] == tiny_tokenizer.assistant_id
    assert window.mask[marker_index] == 1


def test_continuation_first_target_and_eos(tiny_tokenizer):
    answer = " ".join(f"phrase{n}" for n in range(2000))
    record = conversation(
        ("user", "Continue."),
        ("assistant", answer),
    )
    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    assert len(windows) > 1

    for previous, current in zip(windows, windows[1:], strict=False):
        first_target = current.prefix_len
        assert current.tokens[first_target] == encoded.ids[current.target_start]
        assert current.mask[first_target] == 1
        assert current.target_start == previous.target_end

    last = windows[-1]
    last_target_index = last.prefix_len + (last.target_end - last.target_start) - 1
    assert last.tokens[last_target_index] == tiny_tokenizer.eos_id
    assert last.mask[last_target_index] == 1


def test_padding_has_zero_mask_and_pad_token(tiny_tokenizer):
    record = conversation(
        ("user", "Court."),
        ("assistant", "Oui."),
    )
    _, (windows, _) = window_pair(record, tiny_tokenizer)
    window = windows[0]

    target_len = window.target_end - window.target_start
    padding_start = window.prefix_len + target_len

    assert padding_start < len(window.tokens)
    assert all(token == tiny_tokenizer.pad_id for token in window.tokens[padding_start:])
    assert all(value == 0 for value in window.mask[padding_start:])


def test_no_target_token_is_duplicated_in_a_conversation(tiny_tokenizer):
    record = conversation(
        ("user", "Première question."),
        ("assistant", " ".join(f"a{n}" for n in range(800))),
        ("user", "Deuxième question."),
        ("assistant", " ".join(f"b{n}" for n in range(800))),
    )

    encoded, (windows, _) = window_pair(record, tiny_tokenizer)

    covered = []
    for window in windows:
        covered.extend(range(window.target_start, window.target_end))

    expected = []
    for region in encoded.regions:
        expected.extend(range(region.marker, region.end))

    assert covered == expected
    assert len(covered) == len(set(covered))


# ---------------------------------------------------------------------------
# Packer: reproducibility, deduplication, splits, atomicity
# ---------------------------------------------------------------------------


def _sample_records(count: int = 40) -> list[dict]:
    records = []
    for index in range(count):
        records.append(
            conversation(
                ("user", f"XXXMARKER{index:04d}XXX question courte."),
                ("assistant", f"Réponse numéro {index} avec un peu de contenu."),
            )
        )
    return records


def test_prepare_is_deterministic(tmp_path, tiny_tokenizer_path):
    records = _sample_records(30)
    source = write_jsonl(tmp_path / "source.jsonl", records)

    first = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "first",
        val_frac=0.1,
        test_frac=0.1,
    )
    second = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "second",
        val_frac=0.1,
        test_frac=0.1,
    )

    assert first["splits"] == second["splits"]

    for name in ("train", "val", "test"):
        for suffix in ("tokens.bin", "mask.bin"):
            assert (tmp_path / "first" / f"{name}.{suffix}").read_bytes() == (
                tmp_path / "second" / f"{name}.{suffix}"
            ).read_bytes()


def test_deduplication_counts_duplicates_without_writing_a_second_copy(
    tmp_path, tiny_tokenizer_path
):
    record = conversation(
        ("user", "Doublon exact."),
        ("assistant", "Réponse identique."),
    )
    source = write_jsonl(tmp_path / "source.jsonl", [record, record])

    meta = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0,
        test_frac=0,
    )

    stats = meta["source_stats"]["source.jsonl"]
    assert stats["examined"] == 2
    assert stats["accepted"] == 1
    assert stats["duplicates"] == 1
    assert meta["splits"]["train"]["conversations"] == 1


def test_split_is_deterministic_and_matches_stable_split(
    tmp_path, tiny_tokenizer_path
):
    records = _sample_records(120)
    source = write_jsonl(tmp_path / "source.jsonl", records)

    meta = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0.2,
        test_frac=0.2,
    )

    expected = {"train": 0, "val": 0, "test": 0}
    for record in records:
        messages = normalize_messages(record["messages"])
        expected[stable_split(messages, 0.2, 0.2)] += 1

    for split, count in expected.items():
        assert meta["splits"][split]["conversations"] == count


def test_no_conversation_leaks_between_splits(tmp_path, tiny_tokenizer_path, tiny_tokenizer):
    records = _sample_records(150)
    source = write_jsonl(tmp_path / "source.jsonl", records)

    prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0.2,
        test_frac=0.2,
    )

    found = {}
    for split in ("train", "val", "test"):
        tokens, _ = read_split(tmp_path / "packed", split)
        text = tiny_tokenizer.decode(tokens.reshape(-1).tolist())

        found[split] = {
            index
            for index in range(len(records))
            if f"XXXMARKER{index:04d}XXX" in text
        }

    assert not (found["train"] & found["val"])
    assert not (found["train"] & found["test"])
    assert not (found["val"] & found["test"])
    assert found["train"] | found["val"] | found["test"] == set(range(len(records)))


def test_binary_dimensions_and_dtypes(tmp_path, tiny_tokenizer_path):
    records = _sample_records(20)
    source = write_jsonl(tmp_path / "source.jsonl", records)

    meta = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0,
        test_frac=0,
    )

    sequence_len = meta["sequence_len"]
    sequences = meta["splits"]["train"]["sequences"]

    token_path = tmp_path / "packed" / "train.tokens.bin"
    mask_path = tmp_path / "packed" / "train.mask.bin"

    assert token_path.stat().st_size == sequences * sequence_len * 2
    assert mask_path.stat().st_size == sequences * sequence_len * 1

    tokens, masks = read_split(tmp_path / "packed", "train")
    assert tokens.dtype == np.uint16
    assert masks.dtype == np.uint8
    assert tokens.shape == (sequences, sequence_len)
    assert masks.shape == (sequences, sequence_len)
    assert set(np.unique(masks)).issubset({0, 1})
    assert np.all(masks[:, 0] == 0)


def test_atomic_write_leaves_no_staging_directory(tmp_path, tiny_tokenizer_path):
    source = write_jsonl(tmp_path / "source.jsonl", _sample_records(10))

    prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0,
        test_frac=0,
    )

    assert (tmp_path / "packed" / "meta.json").is_file()
    assert list(tmp_path.glob(".packed.tmp-*")) == []


def test_staging_is_cleaned_up_after_failure(
    tmp_path, tiny_tokenizer_path, monkeypatch
):
    import tinyllm.sft_data as sft_data

    source = write_jsonl(tmp_path / "source.jsonl", _sample_records(5))

    def explode(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(sft_data, "_write_window", explode)

    with pytest.raises(RuntimeError, match="simulated"):
        prepare(
            [source],
            tiny_tokenizer_path,
            tmp_path / "packed",
            val_frac=0,
            test_frac=0,
        )

    assert not (tmp_path / "packed").exists()
    assert list(tmp_path.glob(".packed.tmp-*")) == []


def test_refuses_to_overwrite_existing_dataset(tmp_path, tiny_tokenizer_path):
    source = write_jsonl(tmp_path / "source.jsonl", _sample_records(2))
    (tmp_path / "packed").mkdir()
    (tmp_path / "packed" / "keep.txt").write_text("do not delete")

    with pytest.raises(FileExistsError):
        prepare(
            [source],
            tiny_tokenizer_path,
            tmp_path / "packed",
            val_frac=0,
            test_frac=0,
        )

    assert (tmp_path / "packed" / "keep.txt").read_text() == "do not delete"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_statistics_match_binary_files(tmp_path, tiny_tokenizer_path):
    records = _sample_records(50)
    source = write_jsonl(tmp_path / "source.jsonl", records)

    meta = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0.2,
        test_frac=0.2,
    )

    sequence_len = meta["sequence_len"]
    total_sequences = 0
    total_assistant = 0

    for split in ("train", "val", "test"):
        stats = meta["splits"][split]
        tokens, masks = read_split(tmp_path / "packed", split)

        assert stats["sequences"] == tokens.shape[0]
        assert stats["tokens"] == tokens.shape[0] * sequence_len
        assert stats["assistant_tokens"] == int(masks[:, 1:].sum())
        assert stats["target_tokens"] == int(masks.sum())
        assert stats["context_tokens"] + stats["target_tokens"] + stats[
            "padding_tokens"
        ] == stats["tokens"]
        assert (
            stats["initial_windows"] + stats["continuation_windows"]
            == stats["sequences"]
        )

        total_sequences += stats["sequences"]
        total_assistant += stats["assistant_tokens"]

    assert meta["totals"]["sequences"] == total_sequences
    assert meta["totals"]["assistant_tokens"] == total_assistant

    accepted = sum(
        meta["source_stats"][name]["accepted"]
        for name in meta["source_stats"]
    )
    assert (
        sum(meta["splits"][split]["conversations"] for split in ("train", "val", "test"))
        == accepted
    )


def test_statistics_count_long_responses_truncated_questions_and_multi_turn(
    tmp_path, tiny_tokenizer_path
):
    long_answer = " ".join(f"long{n}" for n in range(2000))
    long_question = " ".join(f"question{n}" for n in range(400))

    records = [
        conversation(
            ("user", long_question),
            ("assistant", long_answer),
        ),
        conversation(
            ("user", "Premier tour."),
            ("assistant", "Première réponse."),
            ("user", "Second tour."),
            ("assistant", "Seconde réponse."),
        ),
    ]
    source = write_jsonl(tmp_path / "source.jsonl", records)

    meta = prepare(
        [source],
        tiny_tokenizer_path,
        tmp_path / "packed",
        val_frac=0,
        test_frac=0,
    )

    train = meta["splits"]["train"]
    assert train["long_responses"] == 1
    assert train["questions_truncated"] == 1
    assert train["multi_turn_conversations"] == 1
    assert train["responses"] == 3
    assert train["continuation_windows"] > 0
