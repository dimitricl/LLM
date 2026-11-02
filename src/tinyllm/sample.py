import argparse
from pathlib import Path

import mlx.core as mx

from tinyllm.config import RunConfig, load_config
from tinyllm.model import TinyLM
from tinyllm.tokenizer import TokenizerWrapper


def apply_repetition_penalty(
    logits: mx.array, token_ids: list[int], penalty: float
) -> mx.array:
    if penalty <= 1.0 or not token_ids:
        return logits
    ids = mx.array(sorted(set(token_ids)))
    selected = logits[ids]
    penalized = mx.where(selected > 0, selected / penalty, selected * penalty)
    return logits.at[ids].add(penalized - selected)


def filter_logits(
    logits: mx.array, top_k: int | None = None, top_p: float | None = None
) -> mx.array:
    if top_k is not None and top_k > 0:
        keep = min(top_k, logits.shape[-1])
        cutoff = mx.sort(logits)[-keep]
        logits = mx.where(logits < cutoff, -mx.inf, logits)
    if top_p is not None and top_p < 1.0:
        order = mx.argsort(logits)[::-1]
        ordered = logits[order]
        probabilities = mx.softmax(ordered)
        cumulative = mx.cumsum(probabilities)
        drop = mx.concatenate(
            [mx.array([False]), cumulative[1:] - probabilities[1:] > top_p]
        )
        ordered = mx.where(drop, -mx.inf, ordered)
        logits = mx.zeros(logits.shape).at[order].add(ordered)
    return logits


def sample_token(
    logits: mx.array,
    temperature: float,
    key: mx.array,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits))
    logits = filter_logits(logits / temperature, top_k, top_p)
    return int(mx.random.categorical(logits, key=key))


def generate_ids(
    model: TinyLM,
    prompt_ids: list[int],
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.0,
    stop_at_eos: bool = True,
    eos_id: int | None = None,
    seed: int = 0,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if len(prompt_ids) > model.cfg.context_len:
        raise ValueError("prompt is longer than the model context")

    key = mx.random.key(seed)
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)[0, -1, :]
    generated: list[int] = []
    for _ in range(max_tokens):
        if cache[0].length >= model.cfg.context_len:
            break
        logits = apply_repetition_penalty(
            logits, prompt_ids + generated, repetition_penalty
        )
        key, subkey = mx.random.split(key)
        token = sample_token(logits, temperature, subkey, top_k, top_p)
        if stop_at_eos and eos_id is not None and token == eos_id:
            break
        generated.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1, :]
    return prompt_ids + generated


def generate(
    model: TinyLM,
    tokenizer: TokenizerWrapper,
    prompt: str,
    **kwargs,
) -> str:
    return tokenizer.decode(
        generate_ids(
            model,
            tokenizer.encode(prompt),
            eos_id=tokenizer.eos_id,
            **kwargs,
        )
    )


def load_run_model(
    run_dir: str | Path, checkpoint: str = "latest"
) -> tuple[TinyLM, TokenizerWrapper, RunConfig]:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    model = TinyLM(cfg.model)
    checkpoint_path = run_dir / "checkpoints" / checkpoint / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"no model checkpoint at {checkpoint_path}")
    model.load_weights(list(mx.load(str(checkpoint_path)).items()))
    return model, TokenizerWrapper(cfg.data.tokenizer_path), cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model, tokenizer, _ = load_run_model(args.run_dir, args.checkpoint)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
