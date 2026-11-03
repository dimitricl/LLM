import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from tinyllm.sft_data import prepare
from tinyllm.sft_dataset import SFTSequenceDataset
from tinyllm.sft_eval import resolve_context_len
from tinyllm.sft_train import masked_loss, packed_context_len


def _records(count: int = 24) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"Question numéro {index} sur un sujet."},
                {
                    "role": "assistant",
                    "content": f"Réponse numéro {index}, différente des autres.",
                },
            ]
        }
        for index in range(count)
    ]


@pytest.fixture()
def packed(tmp_path, tiny_tokenizer_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in _records())
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "packed"
    meta = prepare(
        [source],
        tiny_tokenizer_path,
        output,
        val_frac=0,
        test_frac=0,
    )

    return output, meta


def _dataset(packed, seed: int = 7) -> SFTSequenceDataset:
    output, meta = packed
    return SFTSequenceDataset(
        output / "train.tokens.bin",
        output / "train.mask.bin",
        meta["context_len"],
        seed,
    )


def test_dataset_shapes_and_dtypes(packed):
    dataset = _dataset(packed)
    batch_size = 4
    tokens, mask = dataset.next_batch(batch_size)

    assert tokens.shape == (batch_size, 513)
    assert mask.shape == (batch_size, 512)
    assert str(tokens.dtype) == "mlx.core.int32"
    assert str(mask.dtype) == "mlx.core.float32"


def test_batch_order_is_reproducible(packed):
    first = _dataset(packed, seed=11).next_batch(8)
    second = _dataset(packed, seed=11).next_batch(8)

    assert np.array_equal(np.array(first[0]), np.array(second[0]))
    assert np.array_equal(np.array(first[1]), np.array(second[1]))


def test_different_seeds_produce_different_order(packed):
    first = np.array(_dataset(packed, seed=1).next_batch(12)[0])
    second = np.array(_dataset(packed, seed=2).next_batch(12)[0])
    assert not np.array_equal(first, second)


def test_state_roundtrip_resumes_identical_stream(packed):
    dataset = _dataset(packed, seed=3)
    dataset.next_batch(8)
    state = dataset.state()

    resumed = _dataset(packed, seed=3)
    resumed.load_state(state)

    expected_tokens, expected_mask = dataset.next_batch(8)
    actual_tokens, actual_mask = resumed.next_batch(8)

    assert np.array_equal(np.array(expected_tokens), np.array(actual_tokens))
    assert np.array_equal(np.array(expected_mask), np.array(actual_mask))


def test_epoch_advances_and_reshuffles(packed):
    dataset = _dataset(packed, seed=0)
    order_before = dataset.order.copy()

    dataset.next_batch(24)
    assert dataset.epoch == 1
    assert not np.array_equal(order_before, dataset.order)

    resumed = _dataset(packed, seed=0)
    resumed.load_state(dataset.state())
    assert np.array_equal(resumed.order, dataset.order)


def test_mask_is_the_stored_mask_without_its_first_column(packed):
    output, meta = packed
    sequence_len = meta["sequence_len"]

    raw_masks = np.fromfile(
        output / "train.mask.bin", dtype=np.uint8
    ).reshape(-1, sequence_len)

    dataset = _dataset(packed)
    state = dataset.state()
    state["cursor"] = 0
    dataset.load_state(state)

    _, mask = dataset.next_batch(1)
    expected = raw_masks[dataset.order[0], 1:].astype(np.float32)

    assert np.array_equal(np.array(mask)[0], expected)


def test_trainer_tensor_contract(packed):
    dataset = _dataset(packed)
    tokens, mask = dataset.next_batch(3)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    assert inputs.shape == (3, 512)
    assert targets.shape == (3, 512)
    assert mask.shape == (3, 512)
    assert np.array_equal(np.array(inputs[0, 0]), np.array(tokens[0, 0]))


def test_rejects_empty_files(tmp_path):
    token_path = tmp_path / "tokens.bin"
    mask_path = tmp_path / "mask.bin"
    token_path.write_bytes(b"")
    mask_path.write_bytes(b"")

    with pytest.raises(ValueError, match="must not be empty"):
        SFTSequenceDataset(token_path, mask_path, context_len=4, seed=0)


def test_rejects_misaligned_token_file(tmp_path):
    token_path = tmp_path / "tokens.bin"
    mask_path = tmp_path / "mask.bin"
    token_path.write_bytes(b"\x00" * 11)
    mask_path.write_bytes(b"\x00" * 6)

    with pytest.raises(ValueError, match="not aligned"):
        SFTSequenceDataset(token_path, mask_path, context_len=4, seed=0)


def test_rejects_token_mask_row_mismatch(tmp_path):
    token_path = tmp_path / "tokens.bin"
    mask_path = tmp_path / "mask.bin"
    sequence_len = 5
    token_path.write_bytes(b"\x00" * sequence_len * 2 * 2)
    mask_path.write_bytes(b"\x00" * sequence_len * 1)

    with pytest.raises(ValueError, match="row mismatch"):
        SFTSequenceDataset(token_path, mask_path, context_len=4, seed=0)


def test_load_state_rejects_incompatible_checkpoint(packed):
    dataset = _dataset(packed, seed=5)
    state = dataset.state()

    other_seed = _dataset(packed, seed=6)
    with pytest.raises(ValueError, match="seed"):
        other_seed.load_state(state)

    wrong_count = dict(state)
    wrong_count["num_sequences"] += 1
    with pytest.raises(ValueError, match="sequence count"):
        _dataset(packed, seed=5).load_state(wrong_count)

    wrong_context = dict(state)
    wrong_context["context_len"] += 1
    with pytest.raises(ValueError, match="context length"):
        _dataset(packed, seed=5).load_state(wrong_context)


def test_hidden_mask_first_column_is_never_a_target(packed):
    output, meta = packed
    sequence_len = meta["sequence_len"]

    raw_masks = np.fromfile(
        output / "train.mask.bin", dtype=np.uint8
    ).reshape(-1, sequence_len)

    assert np.all(raw_masks[:, 0] == 0)
    assert np.all(raw_masks.sum(axis=1) > 0)


class _FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def __call__(self, tokens):
        return self.logits


def test_masked_loss_only_uses_supervised_positions():
    rng = np.random.default_rng(0)
    batch, sequence_len, vocab = 2, 6, 7

    tokens = mx.array(rng.integers(0, vocab, size=(batch, sequence_len)))
    logits = mx.array(rng.normal(size=(batch, sequence_len - 1, vocab)))
    mask = mx.array(
        [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
        dtype=mx.float32,
    )

    got = float(masked_loss(_FixedLogits(logits), tokens, mask))

    logits_np = np.array(logits, dtype=np.float64)
    targets = np.array(tokens)[:, 1:]
    mask_np = np.array(mask, dtype=np.float64)

    shift = logits_np.max(axis=-1, keepdims=True)
    log_probs = logits_np - shift - np.log(
        np.exp(logits_np - shift).sum(axis=-1, keepdims=True)
    )
    picked = np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    expected = float((-(picked) * mask_np).sum() / mask_np.sum())

    assert abs(got - expected) < 1e-5


def test_masked_loss_is_invariant_to_unmasked_logits():
    rng = np.random.default_rng(1)
    tokens = mx.array(rng.integers(0, 5, size=(1, 5)))
    mask = mx.array([[0, 1, 0, 1]], dtype=mx.float32)

    logits_a = np.array(rng.normal(size=(1, 4, 5)), dtype=np.float32)
    logits_b = logits_a.copy()
    logits_b[:, 0, :] += 100.0
    logits_b[:, 2, :] -= 100.0

    loss_a = float(masked_loss(_FixedLogits(mx.array(logits_a)), tokens, mask))
    loss_b = float(masked_loss(_FixedLogits(mx.array(logits_b)), tokens, mask))

    assert loss_a == pytest.approx(loss_b, rel=0, abs=1e-6)


def _write_meta(path, layout: str, context_len: int = 512) -> None:
    (path / "meta.json").write_text(
        json.dumps({
            "sequence_layout": layout,
            "context_len": context_len,
        })
    )
    (path / "tokens.bin").write_bytes(b"")


def test_trainer_and_eval_accept_both_layouts(tmp_path):
    for layout in (
        "independent_conversation_windows",
        "contextual_conversation_windows",
    ):
        missing_ok = tmp_path / layout
        missing_ok.mkdir()
        _write_meta(missing_ok, layout)

        token_path = missing_ok / "tokens.bin"
        assert packed_context_len(token_path, None, 512) == 512
        assert resolve_context_len(str(token_path), None, 512) == 512


def test_trainer_and_eval_reject_unknown_layout(tmp_path):
    _write_meta(tmp_path, "unknown_layout")

    with pytest.raises(ValueError, match="Unsupported SFT dataset layout"):
        packed_context_len(tmp_path / "tokens.bin", None, 512)

    with pytest.raises(ValueError, match="Unsupported SFT dataset layout"):
        resolve_context_len(str(tmp_path / "tokens.bin"), None, 512)
