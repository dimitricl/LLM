# TinyLLM

Small decoder-only language model (FR/EN) trained from scratch with
[MLX](https://github.com/ml-explore/mlx) on Apple Silicon: tokenizer,
pre-training, SFT, evaluation, quantization and a local chat server.

Home-made model: **V9** (Jarvis v2, tools included; final results in `docs/V9_RESULTS.md` section 8).
Fine-tuned track on LFM3: branch `feat/jarvis-lfm3`.

Current model: **V9**, 86.5 M parameters (16 layers, d_model 640, GQA 10:2,
context 512), pre-trained on ~1.97 B tokens. Details and numbers:
[`docs/V9_RESULTS.md`](docs/V9_RESULTS.md).

## Layout

| Path | Content |
|---|---|
| `src/tinyllm/` | model, training loops, SFT data/packing, decoding, quantization |
| `scripts/` | data builders, evaluation (`chat_regression.py`, `general_probe.py`), `serve.py` |
| `configs/` | run configs (`pretrain_v9.json`, `sft_v8.json`, ...) |
| `tests/` | pytest suite |
| `docs/` | plans, results, distillation and teacher benchmarks |
| `docs/ctx/` | project memory (status, architecture, decisions, todo) |
| `exports/` | int8 / int4 bundles (git-ignored) |

`runs/`, `data/` and `exports/` are git-ignored (checkpoints and corpora).

## Quick start

```bash
uv sync
.venv/bin/python -m pytest -q
python scripts/serve.py --model-dir exports/tinyllm-v9-int8 --port 8080
```

Reproduction commands for pre-training, SFT and evaluation are in
`docs/V9_RESULTS.md` (section 5). Older commands: `COMMANDS.md` (V3-era).
