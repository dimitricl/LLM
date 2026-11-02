import numpy as np

from tinyllm.data import clean_text, dedupe, pack_documents, read_bin, split_documents, text_hash, write_bin


def test_clean_text_strips_html_and_whitespace():
    raw = "  <p>Hello</p>\n\n\n\n   world  \t !  "
    assert clean_text(raw) == "Hello\n\nworld !"


def test_dedupe_keeps_first_occurrence():
    docs = [{"text": "a"}, {"text": "b"}, {"text": "a"}]
    assert [d["text"] for d in dedupe(docs)] == ["a", "b"]


def test_split_is_deterministic_and_disjoint():
    docs = [{"text": f"document number {i} " + "x" * i} for i in range(2000)]
    first = split_documents(docs, val_frac=0.1, test_frac=0.1)
    second = split_documents(docs, val_frac=0.1, test_frac=0.1)
    assert [text_hash(d["text"]) for d in first["val"]] == [
        text_hash(d["text"]) for d in second["val"]
    ]
    assert 100 < len(first["val"]) < 300
    assert 100 < len(first["test"]) < 300
    ids = {name: {text_hash(d["text"]) for d in part} for name, part in first.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


def test_pack_documents_inserts_eos_and_drops_remainder():
    docs = [[1, 2, 3], [4, 5]]
    assert list(pack_documents(docs, context_len=4, eos_id=9)) == [[1, 2, 3, 9]]
    assert list(pack_documents(docs, context_len=3, eos_id=9)) == [[1, 2, 3], [9, 4, 5]]


def test_pack_documents_all_chunks_have_context_len():
    docs = [[i] * 7 for i in range(5)]
    chunks = list(pack_documents(docs, context_len=8, eos_id=0))
    assert chunks
    assert all(len(chunk) == 8 for chunk in chunks)


def test_bin_roundtrip(tmp_path):
    path = tmp_path / "train.bin"
    written = write_bin([[1, 2, 3], [4, 5, 6]], path)
    assert written == 6
    array = read_bin(path)
    assert array.dtype == np.uint16
    assert array.tolist() == [1, 2, 3, 4, 5, 6]
