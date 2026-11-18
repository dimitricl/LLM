# Datasets (small, synthetic or hand-written)

Only data that this repository generates itself is published here. Large corpora
(pre-training text, downloaded SFT sets, teacher answers) are NOT redistributed because their
licenses were not checked; the scripts that rebuild them are in `scripts/`.

| File | Rows | Origin | Rebuild |
|---|---|---|---|
| `persona_jarvis_v1.jsonl` | 800 | Persona "Jarvis" (one canonical reply per intent) | `scripts/build_persona_data.py` (seed 42, `configs/persona.json`) |
| `tools_v2.jsonl` | 6 000 | `<tool>calc(..)</tool><result>..</result>` examples, results computed by `src/tinyllm/tools.py` | `scripts/build_tools_data.py` (seed 42) |
| `reasoning_v1.jsonl` | 3 000 | Short `<think>` problems with verified answers (7 families) | `scripts/reasoning_gen.py` (seed 11) |
| `conversational_seed.jsonl` | 153 | Hand-written short French conversations | - |

Format: one JSON object per line, `{"messages": [{"role": "user|assistant", "content": ...}]}`.
