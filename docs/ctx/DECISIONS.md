# DECISIONS
- 2026-11-18: V9 tokenizer = V8 tokenizer — reuse packed SFT data without re-tokenization (docs/V9_PLAN.md).
- 2026-11-18: work continues on branch `feat/tinyllm-v9` — V9 files were uncommitted on the V8 branch.
- 2026-11-18: `.bak` copies and empty root files removed (recoverable from git history, commit deec41e).
- 2026-11-18: keep V9 as the home-made reference/portfolio model — user wants to be able to show it (parent-model track compared in parallel).
- 2026-11-18: tools use plain-text format <tool>name(args)</tool><result>..</result> — no vocabulary change needed.
