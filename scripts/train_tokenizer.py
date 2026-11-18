import argparse

from tinyllm.config import load_config
from tinyllm.data import clean_text
from tinyllm.prepare import iter_mixed_documents
from tinyllm.tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-chars", type=int, default=2_000_000_000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    budget = args.max_chars
    consumed = 0

    def texts():
        nonlocal consumed
        for _, document in iter_mixed_documents(cfg.data.sources, cfg.train.seed):
            text = clean_text(document["text"])
            if not text:
                continue
            remaining = budget - consumed
            if remaining <= 0:
                return
            text = text[:remaining]
            consumed += len(text)
            yield text

    train_tokenizer(
        texts(), vocab_size=cfg.model.vocab_size, out_path=cfg.data.tokenizer_path
    )
    print(f"trained tokenizer on ~{consumed} characters")


if __name__ == "__main__":
    main()
