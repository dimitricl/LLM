import argparse
import time

from tinyllm.sample import generate_ids, load_run_model


def encode_turns(turns: list[dict], tokenizer) -> list[int]:
    """Encode completed turns exactly like the SFT dataset does."""
    ids: list[int] = []

    for message in turns:
        if message["role"] == "user":
            ids.append(tokenizer.user_id)
            ids.extend(tokenizer.encode(message["content"]))
        else:
            ids.append(tokenizer.assistant_id)
            ids.extend(tokenizer.encode(message["content"]))
            ids.append(tokenizer.eos_id)

    return ids


def build_prompt(
    turns: list[dict],
    user_text: str,
    tokenizer,
    max_prompt_tokens: int,
) -> list[int]:
    """Build ``... <|user|> question <|assistant|>`` with turn-level trimming.

    The dataset prepends the same special tokens, so the chat prompt matches
    the training distribution.  Older turns are dropped first; if a single
    question still does not fit, it is truncated head-only like the packer.
    """
    user_ids = tokenizer.encode(user_text)
    budget = max_prompt_tokens - 2  # USER marker + ASSISTANT marker

    if len(user_ids) > budget:
        print(
            f"Avertissement : question tronquée à {budget} tokens "
            f"(sur {len(user_ids)})."
        )
        user_ids = user_ids[:budget]

    prompt = encode_turns(turns, tokenizer) + [tokenizer.user_id]
    prompt += user_ids + [tokenizer.assistant_id]

    while len(prompt) > max_prompt_tokens and len(turns) > 1:
        turns = turns[2:] if turns[0]["role"] == "user" else turns[1:]
        prompt = encode_turns(turns, tokenizer) + [tokenizer.user_id]
        prompt += user_ids + [tokenizer.assistant_id]

    if len(prompt) > max_prompt_tokens:
        # Only the current question remains: keep the tail of the prompt.
        prompt = prompt[-(max_prompt_tokens):]

    return prompt


def main():
    parser = argparse.ArgumentParser(description="TinyLLM Chat")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Ne pas afficher la vitesse de génération (tokens/s).",
    )
    args = parser.parse_args()

    print("\nChargement de TinyLLM...")

    model, tokenizer, cfg = load_run_model(
        args.run_dir, args.checkpoint
    )

    print(f"Modèle chargé : {model.num_parameters():,} paramètres")
    print(f"Contexte : {cfg.model.context_len} tokens")
    print(f"Décodage: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print(f"  rep_penalty={args.repetition_penalty}, no_repeat_ngram={args.no_repeat_ngram_size}")
    print("\nTinyLLM Chat")
    print("Commandes : /reset | /exit\n")

    turns: list[dict] = []

    while True:
        try:
            user_input = input("Toi > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nFermeture.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/reset":
            turns.clear()
            print("Historique effacé.\n")
            continue

        max_prompt_tokens = cfg.model.context_len - args.max_tokens
        prompt_ids = build_prompt(
            turns,
            user_input,
            tokenizer,
            max_prompt_tokens,
        )

        remaining = min(
            args.max_tokens,
            cfg.model.context_len - len(prompt_ids),
        )

        if remaining <= 0:
            print("Contexte saturé. Utilise /reset.\n")
            continue

        start = time.perf_counter()
        output_ids = generate_ids(
            model,
            prompt_ids,
            max_tokens=remaining,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop_at_eos=True,
            eos_id=tokenizer.eos_id,
            stop_on_user_token=True,
            user_id=tokenizer.user_id,
            seed=args.seed + len(turns),
        )
        elapsed = time.perf_counter() - start

        answer = tokenizer.decode(output_ids[len(prompt_ids):]).strip()

        print(f"\nTinyLLM > {answer}\n")

        if not args.no_speed:
            generated = len(output_ids) - len(prompt_ids)
            tokens_per_second = generated / elapsed if elapsed > 0 else 0.0
            print(
                f"  [gen] {generated} tokens | {elapsed:.2f} s | "
                f"{tokens_per_second:.1f} tok/s | prompt {len(prompt_ids)} tokens"
            )

        turns.append({"role": "user", "content": user_input})
        turns.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
