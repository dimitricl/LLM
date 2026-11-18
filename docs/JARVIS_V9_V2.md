# V9 Jarvis v2 (from scratch line): tools fixes + programmatic "thinking"

Run: `scripts/run_jarvis_v2.sh` (2026-11-18). 600 steps from `sft_v9_anneal_a`, lr 1e-5,
8 574 sequences (reflexes, persona, 6 000 tool rows, 3 000 reasoning rows), 41 min, val_loss
0.457 -> 0.293. Checkpoint `runs/sft_v9_jarvis2`, bundle `exports/tinyllm-v9-jarvis2`.

## Results (one run each, greedy for reasoning)
| Measure | anneal_a | Jarvis v1 | Jarvis v2 |
|---|---|---|---|
| Neutral bench with tools (45) | 12 (no tools) | 21 | 20 |
| 21-case regression | 21 | 16 | 18 |
| Arithmetic copy test (40 fresh questions) | - | 14 | **28** |
| Date tool (text follows tool result) | - | wrong (20 nov.) | **correct** |
| Reasoning, held-out wording (63) | 0 | 3 | 10 (6 at temp 0 re-run) |
| Reasoning, training wording, new problems (63) | - | - | **41** |

By family on training wording: syllogism 8/9, share 9/9, multiply 8/9, sub_add 7/9,
ordering 4/9, sequence 3/9, time 2/9.

## Conclusions
- Tool fixes worked (date, copying numbers). Remaining defects: wrong arguments on chained
  expressions (`218 - 146 + 3` -> `calc(218+146)`), final sentence sometimes mis-copies the result.
- Programmatic thinking is learned as templates: ~65 % on seen wording, ~10 % on rephrasings.
  It does NOT generalize at 86 M parameters with this data. Do not present it as reasoning.
- Neutral bench is unchanged (20 vs 21): real prompts gain nothing from the thinking data.

## Ideas not done
- Mitigate copy errors at runtime (validate the answer against the tool result).
- Far more varied phrasing (LLM-paraphrased problems) before expecting generalization.
