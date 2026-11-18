# PROJECT_STATUS
Updated: 2026-11-18

## Current state
- Branch `feat/tinyllm-v9` (from `feat/tinyllm-v8-sft`); work committed locally, not pushed.
- V9: 86.5 M params, pretrain done (30 000 steps, val_loss 3.1587, 1.97 B tokens).
- SFT annealing `runs/sft_v9_anneal_a`: 103/105 on the 21-case bench x5 seeds, leaks 0/10.
- `runs/sft_v9_conv` running (PID 27482, 6 000 steps): step 2900 at 14:21, val_loss 2.2269 @2750 (gain ~0.005 per 250 steps, below the 0.01 stop threshold). ETA ~3.5 h [?].
- Exports in `exports/`: `tinyllm-v9-lite` (int8 + fp16 embeddings, 107 MB, 103/105), `tinyllm-v9-tiny` (int4, 71 MB, 90/105), `-air` (int8, 142 MB).
- Parent-model comparison (12 prompts, 2026-11-18, my reading, one run): LFM3-1.7B 4-bit ~10/12 (973 MB MLX active), LFM2.5-230M 8-bit ~10/12, LFM2.5-350M ~7-8/12, V9 lite ~3/12. LFM3 needs `mlx-vlm` (no `mlx-lm` support yet). Env: `/Volumes/Lexar/Projects/llm-eval/` (venv + models, outside repo).
- Backup: `~/backups/tinyllm/pretrain_v9_best` (sha256 verified).
- Tests: 229 pass, 1 fail (`test_train_config_has_max_hours`, 24.0 expected vs 48.0 in config; fails before today's changes too).

## Last session
- Phase 0 prep (no GPU): neutral bench (`scripts/bench_cases.py`, `bench_neutral.py`), tool format (`src/tinyllm/tools.py`, `scripts/build_tools_data.py`), persona data (`configs/persona.json` draft, `scripts/build_persona_data.py`). Datasets generated in `data/sft/{persona_v1,tools_v1}` (git-ignored).
- Created `recap` skill + Haiku agent in `~/.claude`.

## Next steps
- When sft_v9_conv ends: run `bench_neutral.py` and the 21-case bench + probe on final vs `anneal_a`.
- Edit `configs/persona.json` (name, replies) before using the persona data.
- Compare V9+tools vs LFM2.5/LFM3 (LoRA persona) on the neutral bench, RAM and speed; then decide.
