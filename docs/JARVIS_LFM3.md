# Jarvis on LFM3-1.7B (fine-tuned track)

Fine-tuned model, NOT trained from scratch. The from-scratch model is V9 (`docs/V9_RESULTS.md`).
Base: `LiquidAI/LFM3-1.7B-MLX-4bit` (LFM Open License v1.0: free under $10M revenue, derivatives stay
under the license, attribution clause not verified). LFM3 loads only through `mlx-vlm` (>= 0.7.11).

## Recipe (2026-11-18)
1. `scripts/lfm/gen_replay.py`: the base model answers 500 general prompts (self-distilled replay).
2. `scripts/lfm/build_lora_mix.py`: 110 unique persona rows (18 %) + replay -> train/valid.
3. LoRA: `python -m mlx_vlm.train --model <dir> --data <mix> --train-type lora --iters 300
   --batch-size 2 --gradient-accumulation-steps 2 --grad-checkpoint --learning-rate 5e-5
   --max-seq-length 512 --train-on-completions --output-path <adapters>` (~7 min).
   Batch 4 without grad checkpointing thrashed on a 17 GB Mac (swap).
4. Eval: `scripts/bench_neutral.py --backend mlx_vlm --model <dir> --adapter <adapters>`.

## Results (one run, regex scoring)
| | Base LFM3-1.7B 4-bit | + Jarvis LoRA |
|---|---|---|
| Neutral bench (45) | 42 | 41 |
| MLX memory (active / peak) | 973 / 1113 MB | 1006 / 1178 MB |

Persona learned (identity stable over multi-turn, refuses live info). Only new failure: "Comment
ça va ?" answered with the identity sentence (leak). Other misses are shared with the base model
(photosynthesis answered in Spanish, yes/no answer too long) or regex false negatives ("Hey!").

## Known issues / TODO
- The canned capabilities reply says "je calcule avec un outil": false for this model (no tool runtime).
  Use a persona variant without it.
- Humor is not learned (persona data has fixed replies only); jokes are generic.
- Language flake (Spanish) on one prompt, present in the base model too.
- Not yet measured on the MacBook Air: speed and real RAM at long context.
- Adapters and venv live outside the repo (`/Volumes/Lexar/Projects/llm-eval/`).
