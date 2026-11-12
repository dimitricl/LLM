
from collections.abc import Iterable
from pathlib import Path

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

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

# The v8 tokenizer uses a larger vocabulary because the v7 tokenizer spent
# 2.73 tokens per French word (accents were split into two byte tokens).
V8_VOCAB_SIZE = 24_576

# Split the text into word/number/punctuation units, keeping any leading
# whitespace attached to the *following* unit.  Trailing punctuation is kept
# inside the word unit (``France.``, ``monde,``) so BPE can learn a single
# token for the frequent French pattern "word + final punctuation".  This is
# the key difference with the v7 tokenizer, where a standalone ``Ġ`` token was
# emitted before almost every French word.  ``ByteLevel`` then only maps
# bytes: it must not split again (``use_regex=False``), otherwise the
# whitespace would become a token on its own.
V8_SPLIT_PATTERN = (
    r"\s*\p{L}+[^\p{L}\p{N}\s]*"
    r"|\s*\p{N}+[^\p{L}\p{N}\s]*"
    r"|\s*[^\p{L}\p{N}\s]+"
)

_SPACE_MARKER = "Ġ"


def build_v8_pre_tokenizer() -> pre_tokenizers.Sequence:
    """Return the NFC-aware, space-attached pre-tokenizer for v8."""
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(V8_SPLIT_PATTERN),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            ),
        ]
    )


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


def _add_missing_space_prefix_tokens(tokenizer: Tokenizer) -> int:
    """Add ``Ġ<byte>`` fallback tokens so a space is never emitted alone.

    BPE occasionally has no merge for ``Ġ`` + the first byte of a rare
    character (``Œ``, ligatures, non-Latin scripts).  The encoder wrapper uses
    these fallback tokens to rewrite ``[Ġ, token]`` as ``[Ġ + first_byte,
    rest_of_bytes]``, which preserves the exact bytes while guaranteeing that
    no standalone ``Ġ`` token is produced before a word.
    """
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    missing = [
        _SPACE_MARKER + char
        for char in alphabet
        if tokenizer.token_to_id(_SPACE_MARKER + char) is None
    ]
    if missing:
        tokenizer.add_tokens(missing)
    return len(missing)


def train_tokenizer_v8(
    texts: Iterable[str],
    vocab_size: int = V8_VOCAB_SIZE,
    out_path: str | Path = "data/v8/tokenizer.json",
) -> Tokenizer:
    """Train the v8 byte-level BPE tokenizer.

    Differences with :func:`train_tokenizer`:

    * Unicode NFC normalisation, so ``e`` + combining acute and the
      precomposed ``é`` share the same tokenization;
    * a word-level split that attaches whitespace to the next unit instead
      of emitting isolated ``Ġ`` tokens;
    * a larger default vocabulary (24 576) for French/English/code.

    The special tokens and their order are unchanged, so ``eos_id``,
    ``pad_id``, ``user_id`` and ``assistant_id`` keep the same roles.
    """
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("vocab_size is smaller than the special-token set.")

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = build_v8_pre_tokenizer()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    _add_missing_space_prefix_tokens(tokenizer)

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

        self._space_marker_id = self._tok.token_to_id(_SPACE_MARKER)
        # The v8 tokenizer is the one with an NFC normalizer.  Only v8 gets
        # the standalone-space rewrite; v7 encoding stays byte-for-byte
        # identical for the v7/v8 comparison.
        self.fixes_standalone_spaces = (
            self._tok.normalizer is not None
            and self._space_marker_id is not None
        )

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
        ids = self._tok.encode(
            text,
            add_special_tokens=False,
        ).ids
        if self.fixes_standalone_spaces:
            ids = self._fix_space_tokens(ids)
        return ids

    def _fix_space_tokens(self, ids: list[int]) -> list[int]:
        """Rewrite ``[Ġ, token]`` pairs so no standalone ``Ġ`` is emitted."""
        for _ in range(8):
            fixed = self._fix_space_tokens_once(ids)
            if fixed == ids:
                break
            ids = fixed
        return ids

    def _fix_space_tokens_once(self, ids: list[int]) -> list[int]:
        output: list[int] = []
        index = 0
        changed = False

        while index < len(ids):
            token_id = ids[index]

            if token_id != self._space_marker_id or index + 1 >= len(ids):
                output.append(token_id)
                index += 1
                continue

            next_id = ids[index + 1]

            if next_id in self._special_ids:
                output.append(token_id)
                index += 1
                continue

            token = self._tok.id_to_token(next_id)
            if not token:
                output.append(token_id)
                index += 1
                continue

            first, rest = token[0], token[1:]
            combined_id = self._tok.token_to_id(_SPACE_MARKER + first)
            if combined_id is None:
                output.append(token_id)
                index += 1
                continue

            rest_ids: list[int] = []
            for char in rest:
                char_id = self._tok.token_to_id(char)
                if char_id is None:
                    rest_ids = []
                    break
                rest_ids.append(char_id)

            if rest and not rest_ids:
                output.append(token_id)
                index += 1
                continue

            output.append(combined_id)
            output.extend(rest_ids)
            index += 2
            changed = True

        return output if changed else ids

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

