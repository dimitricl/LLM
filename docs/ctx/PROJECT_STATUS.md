# PROJECT_STATUS
Updated: 2026-11-18

## Current state
- Branch `feat/tinyllm-v9` (from `feat/tinyllm-v8-sft`); V9 work committed, not pushed.
- V9: 86.5 M params, pretrain done (30 000 steps, val_loss 3.1587, 1.97 B tokens).
- SFT annealing `runs/sft_v9_anneal_a`: 103/105 on 21-case bench x5 seeds, leaks 0/10 (V8: 1/10).
- `runs/sft_v9_conv` running (PID 27482, 6 000 steps): step 1700 at 2026-11-18 12:55, val_loss 2.272 @1500, monotone decrease.
- Exports int8/int4 in `exports/`; `scripts/serve.py` = OpenAI-compatible server.
- Tests: 211 pass, 1 fail (`tests/test_train_v8.py::test_train_config_has_max_hours`: expects 24.0, config has 48.0; fails before the 2026-11-18 cleanup too).
- Backup: `~/backups/tinyllm/pretrain_v9_best` (model, optimizer, state, config, tokenizer; sha256 verified).

## Last session
- Audited project, committed V9 files, removed stray files and `.bak`, wrote README and docs/ctx.

## Next steps
- Evaluate `sft_v9_conv` (step_1500 and final) with `chat_regression.py` and `general_probe.py`.
- Measure plan go/no-go metrics not yet measured: loop_rate, OOV FR, correct refusals, false refusals.
- Dataset v5 (ACK, PERSONAL_INFO pools, short identity), then 300-step annealing.
- Fix or update the failing max_hours test.
