from pathlib import Path

import pytest

from tinyllm.tokenizer import (
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    USER_TOKEN,
    TokenizerWrapper,
    build_v8_pre_tokenizer,
    train_tokenizer_v8,
)

CORPUS = [
    (
        "Bonjour, ceci est un petit corpus de test pour la tokenisation. "
        "L'assistant répond en français avec des phrases simples. "
        "La France est un pays d'Europe et sa capitale est Paris. "
        "é è à ç ù â ê î ô û ë ï ü. "
    ) * 80,
    (
        "The user asks a question and the assistant gives a short answer. "
        "Conversation history helps the model follow instructions. "
    ) * 80,
    (
        "def hello_world(value):\n"
        "    return value + 1\n\n"
        "for index in range(10):\n"
        "    print(hello_world(index))\n"
    ) * 40,
]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> TokenizerWrapper:
    path = tmp_path_factory.mktemp("tokenizer_v8") / "tokenizer.json"
    train_tokenizer_v8(CORPUS, vocab_size=512, out_path=path)
    return TokenizerWrapper(path)


def test_special_tokens_keep_their_roles(tokenizer):
    assert tokenizer.eos_id == tokenizer._tok.token_to_id(EOS_TOKEN)
    assert tokenizer.pad_id == tokenizer._tok.token_to_id(PAD_TOKEN)
    assert tokenizer.user_id == tokenizer._tok.token_to_id(USER_TOKEN)
    assert tokenizer.assistant_id == tokenizer._tok.token_to_id(ASSISTANT_TOKEN)
    for token in SPECIAL_TOKENS:
        assert tokenizer._tok.token_to_id(token) is not None


def test_roundtrip_is_lossless(tokenizer):
    text = "Bonjour, comment vas-tu ? La France est belle. 123"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_nfc_normalisation_shares_tokenization(tokenizer):
    decomposed = "e\u0301"
    precomposed = "é"
    assert tokenizer.encode(decomposed) == tokenizer.encode(precomposed)


def test_pre_tokenizer_attaches_space_to_next_word():
    pre_tokenizer = build_v8_pre_tokenizer()
    pieces = [piece for piece, _ in pre_tokenizer.pre_tokenize_str("bonjour monde")]
    assert pieces == ["bonjour", "Ġmonde"]
    assert all(piece != "Ġ" for piece in pieces)


def test_no_standalone_space_before_words(tokenizer):
    text = "La France est un pays d'Europe. Sa capitale est Paris."
    pieces = [piece for piece, _ in tokenizer._tok.pre_tokenizer.pre_tokenize_str(text)]
    assert all(piece != "Ġ" for piece in pieces)
    assert all(
        not piece.startswith("Ġ") or len(piece) > 1
        for piece in pieces[1:]
    )


def test_no_cross_word_merge_in_vocabulary(tokenizer):
    for token, _ in tokenizer._tok.get_vocab().items():
        # Leading whitespace markers (space ``Ġ``, newline ``Ċ``, tab ``ĉ``)
        # are legitimate indentation prefixes.  A space marker *after* a
        # non-whitespace character would mean a merge across two words.
        body = token.lstrip("ĠĊĉ")
        assert "Ġ" not in body, token


def test_encoder_never_emits_standalone_space_before_word(tokenizer):
    assert tokenizer.fixes_standalone_spaces
    text = "Œuvres œuvres Œil à côté Élève Çà et là"
    tokens = [
        tokenizer._tok.id_to_token(token_id)
        for token_id in tokenizer.encode(text)
    ]
    assert "Ġ" not in tokens


def test_space_fallback_preserves_bytes(tokenizer):
    text = " Œuvres"
    ids = tokenizer.encode(text)
    total_bytes = sum(tokenizer.token_bytes(token_id) for token_id in ids)
    assert total_bytes == len(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text
