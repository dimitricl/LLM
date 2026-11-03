#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train_tokenizer.py --config configs/phase1_tinystories.json --max-chars 20000000
uv run python -m tinyllm.prepare --config configs/phase1_tinystories.json
uv run python -m tinyllm.train --config configs/phase1_tinystories.json
