
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = ""
PAD_TOKEN = "<|pad|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

SPECIAL_TOKENS = [
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
]


def _bytelevel_inverse() -> dict[str, int]:
    """Map the ByteLevel printable alphabet back to source bytes."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), 256))

    codepoints = byte_values.copy()
    extra = 0

    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1

    return {
        chr(codepoint): value
        for value, codepoint in zip(byte_values, codepoints)
    }


_BYTELEVEL_INVERSE = _bytelevel_inverse()


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 16384,
    out_path: str | Path = "data/tokenizer.json",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))

    return tokenizer


class TokenizerWrapper:
    def __init__(self, path: str | Path) -> None:
        self._tok = Tokenizer.from_file(str(path))

        self.eos_id = self._get_special_id(EOS_TOKEN)
        self.pad_id = self._get_special_id(PAD_TOKEN)
        self.user_id = self._get_special_id(USER_TOKEN)
        self.assistant_id = self._get_special_id(ASSISTANT_TOKEN)

        self._special_ids = set()

        for token in SPECIAL_TOKENS:
            special_id = self._tok.token_to_id(token)
            if special_id is not None:
                self._special_ids.add(special_id)

    def _get_special_id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)

        if token_id is None:
            raise ValueError(
                f"Tokenizer is missing special token {token!r}."
            )

        return token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(
            text,
            add_special_tokens=False,
        ).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(
            ids,
            skip_special_tokens=False,
        )

    def token_bytes(self, token_id: int) -> int:
        if token_id in self._special_ids:
            return 0

        token = self._tok.id_to_token(token_id)

        if token is None:
            raise ValueError(f"Unknown token id: {token_id}")

        try:
            raw_bytes = bytes(
                _BYTELEVEL_INVERSE[char]
                for char in token
            )
        except KeyError as exc:
            raise ValueError(
                "Token is not encoded with the expected ByteLevel alphabet."
            ) from exc

        return len(raw_bytes)

