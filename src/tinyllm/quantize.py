import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.tokenizer import TokenizerWrapper


def _effective_group_size(model: nn.Module, requested: int) -> int:
    dimensions = [
        parameter.shape[-1]
        for _, parameter in tree_flatten(model.parameters())
        if parameter.ndim == 2
    ]
    compatible = math.gcd(*dimensions) if dimensions else requested
    group_size = min(requested, compatible)
    if group_size <= 0:
        raise ValueError("group_size must be compatible with linear input dimensions")
    return group_size


def quantize_linears(
    model: nn.Module, group_size: int = 64, bits: int = 4
) -> nn.Module:
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, or 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    group_size = _effective_group_size(model, group_size)
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda path, module: isinstance(module, nn.Linear),
    )
    return model


def save_quantized(
    model: TinyLM,
    cfg: ModelConfig,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    bits: int = 4,
    group_size: int = 64,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(weights)
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    metadata = {
        "model": asdict(cfg),
        "quantization": {"bits": bits, "group_size": group_size},
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")
    size = (out_dir / "model.safetensors").stat().st_size
    return {"size_bytes": size, "size_mb": size / 2**20}


def load_model(
    model_dir: str | Path,
) -> tuple[TinyLM, TokenizerWrapper]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "config.json").read_text())
    cfg = ModelConfig(**metadata["model"])
    quantization = metadata["quantization"]
    model = quantize_linears(
        TinyLM(cfg),
        group_size=quantization["group_size"],
        bits=quantization["bits"],
    )
    weights_path = model_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"no quantized weights at {weights_path}")
    model.load_weights(list(mx.load(str(weights_path)).items()))
    return model, TokenizerWrapper(model_dir / "tokenizer.json")


def main() -> None:
    from tinyllm.eval import evaluate_bpb
    from tinyllm.sample import load_run_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--val-bin")
    parser.add_argument("--max-degradation", type=float, default=0.15)
    args = parser.parse_args()

    model, tokenizer, cfg = load_run_model(args.run_dir, args.checkpoint)
    before = None
    if args.val_bin:
        before = evaluate_bpb(model, args.val_bin, tokenizer, cfg.model.context_len)
    quantized = quantize_linears(model, args.group_size, args.bits)
    report = save_quantized(
        quantized,
        cfg.model,
        cfg.data.tokenizer_path,
        args.out,
        bits=args.bits,
        group_size=args.group_size,
    )
    if before is not None:
        after = evaluate_bpb(
            quantized, args.val_bin, tokenizer, cfg.model.context_len
        )
        degradation = after["bpb"] / before["bpb"] - 1.0
        report.update(
            {
                "bpb_before": before["bpb"],
                "bpb_after": after["bpb"],
                "degradation": degradation,
            }
        )
        if degradation > args.max_degradation:
            raise SystemExit(
                f"BPB degradation {degradation:.1%} exceeds "
                f"{args.max_degradation:.1%}"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
