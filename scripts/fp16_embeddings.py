#!/usr/bin/env python3
"""Cast every float32 tensor of a quantized safetensors file to float16.

For int8/int4 bundles this shrinks embeddings, scales and norms (~36 MB less on V9).

    python scripts/fp16_embeddings.py in.safetensors out.safetensors
"""

from __future__ import annotations

import sys

import mlx.core as mx


def main() -> None:
    source, target = sys.argv[1], sys.argv[2]
    weights = mx.load(source)
    weights = {
        k: (v.astype(mx.float16) if v.dtype == mx.float32 else v) for k, v in weights.items()
    }
    mx.save_safetensors(target, weights)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
