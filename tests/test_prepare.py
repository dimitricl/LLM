import json
from pathlib import Path

from tinyllm.config import DataConfig, RunConfig
from tinyllm.data import read_bin
from tinyllm.prepare import prepare_data
from tinyllm.tokenizer import train_tokenizer

TEXTS = [f"Document {i}. " + "the quick brown fox jumps over the lazy dog " * (i % 5 + 1)
         for i in range(120)]


def _setup(tmp_path: Path) -> RunConfig:
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(TEXTS, vocab_size=320, out_path=tokenizer_path)
    return RunConfig(
        run_name="test",
        output_dir=str(tmp_path / "runs"),
        data=DataConfig(
            tokenizer_path=str(tokenizer_path),
            train_bin=str(tmp_path / "train.bin"),
            val_bin=str(tmp_path / "val.bin"),
            test_bin=str(tmp_path / "test.bin"),
        ),
    )


def _stream():
    for i, text in enumerate(TEXTS):
        yield "test", {"id": str(i), "text": text}


def test_prepare_data_writes_bins_and_meta(tmp_path):
    cfg = _setup(tmp_path)
    meta = prepare_data(cfg, document_stream=_stream, max_docs=120)
    assert meta["tokens"]["train"] > 0
    assert Path(cfg.data.train_bin).stat().st_size % 2 == 0
    on_disk = json.loads(Path(cfg.data.train_bin).with_name("meta.json").read_text())
    assert on_disk["context_len"] == cfg.model.context_len
    assert len(read_bin(cfg.data.train_bin)) % cfg.model.context_len == 0


def test_prepare_data_is_idempotent(tmp_path):
    cfg = _setup(tmp_path)
    first = prepare_data(cfg, document_stream=_stream, max_docs=120)
    second = prepare_data(cfg, document_stream=_stream, max_docs=120)
    assert first == second
