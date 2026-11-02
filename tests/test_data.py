from tinyllm.data import clean_text, dedupe, split_documents, text_hash


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
