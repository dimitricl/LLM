# DECISIONS
- 2026-11-18: V9 tokenizer = V8 tokenizer — reuse packed SFT data without re-tokenization (docs/V9_PLAN.md).
- 2026-11-18: work continues on branch `feat/tinyllm-v9` — V9 files were uncommitted on the V8 branch.
- 2026-11-18: `.bak` copies and empty root files removed (recoverable from git history, commit deec41e).
- 2026-11-18: keep V9 as the home-made reference/portfolio model — user wants to be able to show it (parent-model track compared in parallel).
- 2026-11-18: tools use plain-text format <tool>name(args)</tool><result>..</result> — no vocabulary change needed.
- 2026-11-18: sft_v9_conv stopped at step 3200 — val_loss plateau (-0.004 per 250 steps) and the conv checkpoint scored 11/45 on the neutral bench vs 12/45 for anneal_a; resumable from step_3000.
- 2026-11-18: persona = Jarvis (objective, funny but serious) — user choice.
- 2026-11-18: parent-model track compared on the neutral bench: LFM3-1.7B 4-bit 42/45 vs V9 ~12/45 — V9 not competitive on facts/calcul/reasoning.
