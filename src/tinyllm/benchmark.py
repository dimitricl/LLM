import argparse
import json
import resource
import time

import mlx.core as mx

from tinyllm.sample import load_run_model, sample_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--prompt",
        default="Once upon a time, a little robot discovered"
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Chargement
    start = time.perf_counter()
    model, tokenizer, cfg = load_run_model(
        args.run_dir, args.checkpoint
    )
    mx.eval(model.parameters())
    load_time = time.perf_counter() - start

    # 2. Préparation du prompt
    prompt_ids = tokenizer.encode(args.prompt)

    if not prompt_ids:
        raise ValueError("Prompt vide")

    if len(prompt_ids) > model.cfg.context_len:
        raise ValueError("Prompt trop long")

    # 3. Prefill
    cache = model.make_cache()
    mx.eval(cache[0].length) if False else None

    start = time.perf_counter()
    logits = model(
        mx.array([prompt_ids], dtype=mx.int32),
        cache=cache
    )[0, -1, :]
    mx.eval(logits)
    prefill_time = time.perf_counter() - start

    # 4. Decode
    generated = []
    key = mx.random.key(args.seed)

    start = time.perf_counter()

    for _ in range(args.max_tokens):
        if cache[0].length >= model.cfg.context_len:
            break

        key, subkey = mx.random.split(key)

        if args.temperature <= 0:
            token = int(mx.argmax(logits))
        else:
            token = sample_token(
                logits,
                args.temperature,
                subkey
            )

        if tokenizer.eos_id is not None and token == tokenizer.eos_id:
            break

        generated.append(token)

        logits = model(
            mx.array([[token]], dtype=mx.int32),
            cache=cache
        )[0, -1, :]

        # Force execution to avoid measuring only queued operations.
        mx.eval(logits)

    decode_time = time.perf_counter() - start

    # 5. Résultats
    generated_text = tokenizer.decode(generated)
    total_time = load_time + prefill_time + decode_time

    max_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    # macOS reports ru_maxrss in bytes.
    max_rss_mb = max_rss / (1024 * 1024)

    results = {
        "checkpoint": args.checkpoint,
        "parameters": model.num_parameters(),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "load_seconds": round(load_time, 4),
        "prefill_seconds": round(prefill_time, 4),
        "decode_seconds": round(decode_time, 4),
        "decode_tokens_per_second": round(
            len(generated) / decode_time, 2
        ) if decode_time > 0 else None,
        "total_seconds": round(total_time, 4),
        "max_rss_mb": round(max_rss_mb, 2),
        "generated_text": generated_text,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
