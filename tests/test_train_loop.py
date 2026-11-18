import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from tinyllm import train as train_module
from tinyllm.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from tinyllm.data import write_bin
from tinyllm.train import rotate_checkpoints, train


def _cfg(tmp_path: Path, name: str, max_steps: int = 6) -> RunConfig:
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    if not train_bin.exists():
        write_bin(
            [[(i + j) % 300 for j in range(40)] for i in range(0, 2000, 40)],
            train_bin,
        )
        write_bin(
            [[(i + j) % 300 for j in range(40)] for i in range(2000, 2400, 40)],
            val_bin,
        )
    return RunConfig(
        run_name=name,
        output_dir=str(tmp_path / "runs"),
        model=ModelConfig(
            vocab_size=320, d_model=32, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_hidden=64, context_len=16,
        ),
        train=TrainConfig(
            micro_batch=2, grad_accum=2, lr=1e-2, max_steps=max_steps,
            eval_interval=3, checkpoint_interval=3, log_interval=1,
            dtype="float32", max_hours=1,
        ),
        data=DataConfig(
            tokenizer_path="", train_bin=str(train_bin), val_bin=str(val_bin)
        ),
    )


def test_train_writes_checkpoint_and_logs(tmp_path):
    cfg = _cfg(tmp_path, "smoke")
    train(cfg)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    meta = json.loads((run_dir / "checkpoints/latest/state.json").read_text())
    assert meta["step"] == 6
    records = [
        json.loads(line)
        for line in (run_dir / "log.jsonl").read_text().splitlines()
    ]
    assert len(records) == 6
    assert all(np.isfinite(record["loss"]) for record in records)
    assert (run_dir / "config.json").exists()


def test_resume_matches_uninterrupted_run(tmp_path):
    train(_cfg(tmp_path, "part", max_steps=3))
    part = mx.load(str(tmp_path / "runs/part/checkpoints/latest/model.safetensors"))
    train(
        _cfg(tmp_path, "resumed", max_steps=3),
        resume=tmp_path / "runs/part/checkpoints/latest",
    )
    resumed_part = mx.load(str(tmp_path / "runs/resumed/checkpoints/latest/model.safetensors"))
    assert part.keys() == resumed_part.keys()
    assert all(mx.allclose(part[key], resumed_part[key], atol=1e-5) for key in part)


def test_grad_accumulation_peak_memory_does_not_scale_linearly(tmp_path, monkeypatch):
    peaks = []
    original_peak_memory = train_module._peak_memory

    def record_peak_memory():
        peak = original_peak_memory()
        peaks.append(peak)
        return peak

    monkeypatch.setattr(train_module, "_peak_memory", record_peak_memory)
    cfg = _cfg(tmp_path, "accum1", max_steps=1)
    train(cfg)
    peak_accum1 = max(peaks)

    peaks.clear()
    cfg = _cfg(tmp_path, "accum4", max_steps=1)
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "train": cfg.train.__class__(**{**cfg.train.__dict__, "grad_accum": 4}),
        }
    )
    train(cfg)
    peak_accum4 = max(peaks)
    assert peak_accum4 <= peak_accum1 * 1.5


def test_rotate_checkpoints_keeps_last_and_best(tmp_path):
    for step in (1, 2, 3):
        (tmp_path / f"step_{step:07d}").mkdir()
    (tmp_path / "best").mkdir()
    rotate_checkpoints(tmp_path, keep=2)
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["best", "step_0000002", "step_0000003"]
