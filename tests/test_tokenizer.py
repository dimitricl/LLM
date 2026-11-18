from pathlib import Path

from tinyllm.tokenizer import (
    EOS_TOKEN,
    SPECIAL_TOKENS,
    TokenizerWrapper,
    train_tokenizer,
)

CORPUS = [
    "Once upon a time there was a little girl named Lily.",
    "She found a small red ball in the garden and laughed.",
    "Le chat dort sur le tapis. Il rêve de poisson.",
    "def add(a, b):\n    return a + b\n",
] * 40


def _train(tmp_path: Path) -> TokenizerWrapper:
    path = tmp_path / "tokenizer.json"
    train_tokenizer(CORPUS, vocab_size=320, out_path=path)
    return TokenizerWrapper(path)


def test_roundtrip_is_lossless(tmp_path):
    tok = _train(tmp_path)
    text = "Hello, world! Ça va ? 123"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_exist(tmp_path):
    tok = _train(tmp_path)
    assert tok.eos_id == tok._tok.token_to_id(EOS_TOKEN)
    for token in SPECIAL_TOKENS:
        assert tok._tok.token_to_id(token) is not None


def test_token_bytes_matches_utf8_length(tmp_path):
    tok = _train(tmp_path)
    text = "Un garçon mange une pomme."
    total = sum(tok.token_bytes(i) for i in tok.encode(text))
    assert total == len(text.encode("utf-8"))


def test_save_and_reload(tmp_path):
    tok = _train(tmp_path)
    reloaded = TokenizerWrapper(Path(tok_path := tmp_path / "tokenizer.json"))
    assert reloaded.encode("hello") == tok.encode("hello")
    assert tok_path.exists()
