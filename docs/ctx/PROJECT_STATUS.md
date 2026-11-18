# PROJECT_STATUS
Updated: 2026-11-18

## Current state
- Branch `feat/tinyllm-v9`. Published 2026-11-18 on github.com/dimitricl/LLM (public): branches feat/tinyllm-v9, feat/jarvis-lfm3, feat/tinyllm-v8 pushed; 8 model bundles as assets of release `tinyllm-v9-models` (files > 100 MB cannot go through git; git-lfs not installed); small synthetic datasets in `datasets/`; index in `MODELS.md`. Nothing merged into main yet.
- V9: 86.5 M params, pretrain done (val_loss 3.1587, 1.97 B tokens). Kept as the home-made reference / portfolio model.
- `runs/sft_v9_conv` STOPPED by decision at step 3200 (best = step 3000, val_loss 2.2228; gains had fallen to ~0.004 per 250 steps). Resumable with `--resume`.
- Neutral bench (45 cases, `scripts/bench_neutral.py`, one run, regex scoring, results in `/Volumes/Lexar/Projects/llm-eval/results/`):
  LFM3-1.7B 4-bit 42/45 | LFM2.5-350M 8-bit 39 | 230M 8-bit 35 | 350M 4-bit 33 | V9 anneal_a 12 | V9 lite 12 | V9 conv(best) 11 | V9 tiny 9.
- V9 conv (best): long generic answers, loses short conversational reflexes (conversation 1/6 vs 6/6 for anneal_a); its few "faits" passes are lucky regex hits on wrong text.
- Memory (MLX active/peak): V9 lite 108/175 MB; LFM2.5-230M 244/369; 350M-4bit 222/371; 350M-8bit 399/570; LFM3-1.7B-4bit 973/1113. RSS is unreliable on macOS.
- LFM3 loads only via `mlx-vlm` (no `mlx-lm` support yet). Env: `/Volumes/Lexar/Projects/llm-eval/` (venv + models, outside repo). LFM license: free under $10M revenue; derivatives stay under the license; attribution clause unverified.
- Persona chosen: Jarvis, objective and funny but serious (`configs/persona.json`); data in `data/sft/{persona_v1,tools_v1}` (git-ignored).
- Tools: `src/tinyllm/tools.py` + `tool_runtime.py` (tested with a scripted model, not yet with a trained model).
- Backup: `~/backups/tinyllm/pretrain_v9_best`. Tests: 233 pass, 1 fail (`max_hours`, pre-existing).

- LoRA smoke tests (2026-11-18, adapters in `llm-eval/adapters/`, data mix `llm-eval/data/jarvis_mix`):
  - LFM2.5-350M 8-bit via `mlx-lm lora` works (150 iters, lr 1e-4, 8 layers, ~5 min): persona learned but model degraded, bench 39 -> 10/45 (loops, identity reply leaks into unrelated answers). Likely causes [?]: 73% persona rows (near-duplicates), weak teacher replay, lr too high.
  - LFM3-1.7B 4-bit via `python -m mlx_vlm.train` works on text-only JSONL (20 iters, lr 5e-5, ~230 tok/s, peak 3.8 GB); calc/facts preserved, identity not yet learned (undertrained). Not comparable 1:1 with the LFM2.5 run.

- LFM3 + Jarvis LoRA v2 (branch `feat/jarvis-lfm3`, see docs/JARVIS_LFM3.md): neutral bench 41/45 (base 42), persona learned, 1006/1178 MB MLX active/peak.
- V9 Jarvis (2026-11-18): `runs/sft_v9_jarvis` = 500-step annealing of `anneal_a` on persona + tools + short reflexes (`scripts/build_jarvis_v9.py`, data `data/sft/packed_v9_jarvis`, 36 min, val_loss 0.586 -> 0.450). Neutral bench 14/45 raw, **21/45 with tool runtime** (`bench_neutral.py --tools`; anneal_a 12/45); 21-case regression 16/21 (failures = old capability wording + 1 weather turn). Bundle `exports/tinyllm-v9-jarvis` (int8 + fp16 emb, 104 MB) with a tool-aware `server.py` (tested: calc, streaming, markup hidden, ~240-390 tok/s on Mac mini).
  - Known defects: argument copy errors (`45+12` -> `calc(45+11)`, `(12+8)*3` -> `calc(12+8)`), date text ignores the tool result (says 14 February when result is 2026-11-18). Fix idea: French-formatted date in `<result>`, more copy/parenthesis examples, retrain ~36 min.

- V9 Jarvis v2 (`runs/sft_v9_jarvis2`, bundle `exports/tinyllm-v9-jarvis2`, details `docs/JARVIS_V9_V2.md`): date fixed, arithmetic copy 28/40 (v1 14/40), regression 18/21, neutral+tools 20/45 (v1 21). Programmatic `<think>`: 41/63 on seen wording but ~10 % on reformulations (templates, not reasoning).

- Cleanup done 2026-11-18: project folder 72 GB -> 7.2 GB. Kept: `runs/{pretrain_v9,sft_v9_anneal_a,sft_v9_jarvis2}/checkpoints/best` only, `data/v9/pretrain/tokenized` (3.8 GB), `data/v8/tokenizer.json`, small packed sets, `general_fr.jsonl` + seed, all `exports/`. Removed: V8 runs, sft_v9_conv, sft_v9_jarvis, raw/filtered corpora, downloaded SFT sets (re-downloadable via `scripts`). Backups in `~/backups/tinyllm/` (pretrain_v9, sft_v9_anneal_a, sft_v9_jarvis2 bests). Checkpoint CLI defaults now `best` (runs keep no `latest`).

## Next steps
- (done) Rebuild the mix: dedupe persona (~150 rows), <=10% persona, self-distilled replay (base model's own answers on general prompts), lower lr; retrain LFM3 for ~300 iters; re-run bench + multi-turn persona check.
- Re-run neutral bench after fine-tuning; check Jarvis persona over multi-turn; check RAM < 2 GB at long context.
- V9 track (portfolio): annealing on persona+tools data from `anneal_a`, evaluate with bench.
