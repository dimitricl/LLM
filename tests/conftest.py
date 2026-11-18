from pathlib import Path

import pytest

from tinyllm.tokenizer import TokenizerWrapper, train_tokenizer


@pytest.fixture(scope="session")
def tiny_tokenizer_path(tmp_path_factory) -> Path:
    """A small local tokenizer with the project special-token layout.

    Special tokens are the first four ids because ``BpeTrainer`` emits them
    before any learned merge, matching the real tokenizer contract.
    """
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"

    texts = [
        (
            "Bonjour, ceci est un petit corpus de test pour la tokenisation. "
            "L'assistant répond en français avec des phrases simples. "
        )
        * 40,
        (
            "def hello_world():\n"
            "    return 'hello'\n\n"
            "for index in range(10):\n"
            "    print(index)\n"
        )
        * 40,
        (
            "The user asks a question and the assistant gives a detailed answer. "
            "Conversation history helps the model follow instructions. "
        )
        * 40,
    ]

    train_tokenizer(texts, vocab_size=512, out_path=path)
    return path


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_path: Path) -> TokenizerWrapper:
    return TokenizerWrapper(tiny_tokenizer_path)
