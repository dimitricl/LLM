# ARCHITECTURE
Updated: 2026-11-18

- `src/tinyllm/model.py` TinyLM (MLX); `config.py` run/model/train configs; `checkpoint.py`.
- Pretrain: `train.py`, `pretrain_v8.py`; data in `data.py`, `dataset.py`, `prepare.py`.
- SFT: `sft_data.py` (packing, mask), `sft_dataset.py`, `sft_train.py` (replay of pretrain tokens via `--replay-bin`).
- Distillation: `distill.py`, `generate_teacher_{logits,responses}.py`.
- Inference: `decoding.py` (anti-loop), `sample.py`, `chat.py`, `quantize.py`; server `scripts/serve.py`.
- Eval: `scripts/chat_regression.py`, `general_probe.py`, `eval_quality.py`.
- Data builders: `scripts/build_sft_v8.py`, `build_annealing_v8.py`, `filter_pretrain_v8.py`.
- Git-ignored heavy dirs: `runs/`, `data/`, `exports/`.
