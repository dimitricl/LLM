from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = ""
SPECIAL_TOKENS = [EOS_TOKEN, "<|pad|>", "<|startoftext|>", "<|unk|>"]


def train_tokenizer(
    texts: Iterable[str], vocab_size: int = 16384, out_path: str | Path = "data/tokenizer.json"
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
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
        eos_id = self._tok.token_to_id(EOS_TOKEN)
        if eos_id is None:
            raise ValueError(f"tokenizer is missing {EOS_TOKEN}")
        self.eos_id = eos_id
        self._special_ids = {
            token_id
            for token in SPECIAL_TOKENS
            if (token_id := self._tok.token_to_id(token)) is not None
        }

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def token_bytes(self, token_id: int) -> int:
        if token_id in self._special_ids:
            return 0
        return len(self._tok.id_to_token(token_id))
