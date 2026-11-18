# TinyLLM V9 model bundles

Home-made model: 86.5 M parameters, pre-trained from scratch (2 B tokens), 512 tokens of context.
Weights are attached to the GitHub Release `tinyllm-v9-models` (one `.tar.gz` per bundle).
Full-precision `best` checkpoints (pretrain_v9, sft_v9_anneal_a, sft_v9_jarvis2) and the tokenizer are in the Release `tinyllm-v9-checkpoints` (1.5 GB, extract at the repo root with `tar -xzf`). The tokenized pretraining corpus is not published.
Unpack one and run `./run.sh` (needs Python >= 3.11 and Apple Silicon; installs MLX locally).

Neutral bench = `scripts/bench_neutral.py` (45 cases, one run, regex scoring). Detailed results:
`docs/V9_RESULTS.md` (section 8) and `docs/JARVIS_V9_V2.md`.

| Bundle | Origin | Format | Size | Neutral bench | Note |
|---|---|---|---|---|---|
| `tinyllm-v9-jarvis2` | `sft_v9_jarvis2` | int8 + fp16 embeddings | 107 MB | **20** (with tools) | Final model: persona, tools, short thinking |
| `tinyllm-v9-jarvis` | `sft_v9_jarvis` | int8 + fp16 embeddings | 107 MB | 21 (with tools) | v1, date tool wrong |
| `tinyllm-v9-lite` | `sft_v9_anneal_a` | int8 + fp16 embeddings | 107 MB | 12 | Short conversation, no tools |
| `tinyllm-v9-tiny` | `sft_v9_anneal_a` | int4 + fp16 embeddings | 71 MB | 9 | Smallest, lower quality |
| `tinyllm-v9-air` | `sft_v9_anneal_a` | int8 (fp32 embeddings) | 143 MB | not measured | First standalone bundle |
| `tinyllm-v9-int8` | `sft_v9_anneal_a` | int8 (fp32 embeddings) | 143 MB | not measured | Raw export, no server |
| `tinyllm-v9-int4` | `sft_v9_anneal_a` | int4 (fp32 embeddings) | 107 MB | not measured (21-case regression 90/105) | Raw export, no server |
| `tinyllm-v9-conv` | `sft_v9_conv` step 3000 | int8 + fp16 embeddings | 107 MB | 11 | Negative result: loses short reflexes |

Known limits: no reliable factual knowledge, the `<think>` format does not generalize to rephrased
problems (41/63 on seen wording, 6 to 10/63 otherwise), process RAM about 160 MB.
