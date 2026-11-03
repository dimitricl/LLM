import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from tinyllm.model import TinyLM
from tinyllm.sample import generate_ids, load_run_model


PROMPTS = [
    # Français courant
    "Explique simplement ce qu'est Internet.",
    "Pourquoi est-il important de dormir suffisamment ?",
    "Quelle est la différence entre météo et climat ?",
    "Explique le fonctionnement d'un moteur électrique.",

    # Informatique et réseaux
    "Explique la différence entre une adresse IP privée et publique.",
    "À quoi sert le protocole DNS ?",
    "Explique le fonctionnement d'un VLAN avec un exemple concret.",
    "Quelle est la différence entre TCP et UDP ?",

    # Programmation
    "Écris une fonction Python qui vérifie si un nombre est premier.",
    "Explique la différence entre une liste et un tuple en Python.",
    "Écris un script Python qui compte les occurrences de chaque mot dans une phrase.",
    "Explique ce qu'est une API REST et donne un exemple.",

    # Raisonnement logique
    "Paul a deux fois l'âge de Marie. Ensemble, ils ont 36 ans. Quel âge ont-ils ? Explique.",
    "Un train parcourt 120 kilomètres en 2 heures. Quelle est sa vitesse moyenne ?",
    "Tous les serveurs sont des ordinateurs. Certains ordinateurs sont portables. Peut-on conclure que certains serveurs sont portables ? Explique.",
    "J'ai trois interrupteurs à l'extérieur d'une pièce et une ampoule à l'intérieur. Je peux entrer une seule fois dans la pièce. Comment identifier l'interrupteur de l'ampoule ?",

    # Respect des instructions
    "Réponds uniquement par trois mots pour décrire Linux.",
    "Explique le DHCP en exactement deux phrases.",
    "Donne quatre avantages de Git sous forme de liste numérotée.",
    "Explique la différence entre authentification et autorisation en moins de 50 mots.",
]


def build_prompt(tokenizer, user_text: str) -> list[int]:
    """Reproduit exactement le format de sérialisation SFT."""
    return (
        [tokenizer.user_id]
        + tokenizer.encode(user_text)
        + [tokenizer.assistant_id]
    )


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict:
    prompt_ids = build_prompt(tokenizer, prompt)

    if len(prompt_ids) >= model.cfg.context_len:
        raise ValueError(
            f"Prompt trop long ({len(prompt_ids)} tokens). "
            f"Contexte : {model.cfg.context_len}"
        )

    available = model.cfg.context_len - len(prompt_ids)
    limit = min(max_tokens, available)

    start = time.perf_counter()

    output_ids = generate_ids(
        model,
        prompt_ids,
        max_tokens=limit,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_id=tokenizer.eos_id,
        seed=seed,
    )

    elapsed = time.perf_counter() - start

    generated_ids = output_ids[len(prompt_ids):]
    answer = tokenizer.decode(generated_ids).strip()

    return {
        "answer": answer,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "generation_seconds": round(elapsed, 4),
        "tokens_per_second": round(
            len(generated_ids) / elapsed, 2
        ) if elapsed > 0 else 0.0,
    }


def load_models(base_run: Path, base_checkpoint: str, sft_checkpoint: Path):
    print("Chargement du modèle pré-entraîné...")
    base_model, tokenizer, cfg = load_run_model(
        base_run, base_checkpoint
    )
    mx.eval(base_model.parameters())

    print("Chargement du modèle SFT...")
    sft_model = TinyLM(cfg.model)
    weights_path = sft_checkpoint / "model.safetensors"

    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    sft_model.load_weights(list(mx.load(str(weights_path)).items()))
    mx.eval(sft_model.parameters())

    if base_model.num_parameters() != sft_model.num_parameters():
        raise ValueError("Les deux modèles n'ont pas le même nombre de paramètres.")

    return base_model, sft_model, tokenizer, cfg


def warmup(model, tokenizer, seed: int):
    prompt_ids = build_prompt(tokenizer, "Bonjour.")
    generate_ids(
        model,
        prompt_ids,
        max_tokens=8,
        temperature=0.0,
        eos_id=tokenizer.eos_id,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparatif TinyLLM pré-entraîné vs SFT"
    )
    parser.add_argument(
        "--base-run",
        type=Path,
        default=Path("runs/phase1_tinystories"),
    )
    parser.add_argument("--base-checkpoint", default="best")
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=Path("runs/sft_v1/checkpoints/best"),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/sft_benchmark"),
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens doit être supérieur à zéro.")

    base_model, sft_model, tokenizer, cfg = load_models(
        args.base_run,
        args.base_checkpoint,
        args.sft_checkpoint,
    )

    print(f"\nParamètres : {base_model.num_parameters():,}")
    print(f"Contexte : {cfg.model.context_len}")
    print(f"Prompts : {len(PROMPTS)}")
    print(f"Température : 0.0")
    print(f"Tokens maximum : {args.max_tokens}")

    print("\nWarmup du modèle pré-entraîné...")
    warmup(base_model, tokenizer, args.seed)

    print("Warmup du modèle SFT...")
    warmup(sft_model, tokenizer, args.seed)

    results = []

    print("\nDémarrage du benchmark...\n")

    for index, prompt in enumerate(PROMPTS, 1):
        print(f"[{index:02d}/{len(PROMPTS)}] {prompt}")

        base_result = generate_answer(
            base_model,
            tokenizer,
            prompt,
            args.max_tokens,
            args.seed,
        )

        sft_result = generate_answer(
            sft_model,
            tokenizer,
            prompt,
            args.max_tokens,
            args.seed,
        )

        results.append({
            "id": index,
            "category": (
                "francais" if index <= 4 else
                "reseaux" if index <= 8 else
                "programmation" if index <= 12 else
                "raisonnement" if index <= 16 else
                "instructions"
            ),
            "prompt": prompt,
            "pretrained": base_result,
            "sft": sft_result,
        })

        print(f"  Base : {base_result['generation_seconds']} s"
              f" | {base_result['tokens_per_second']} tok/s")
        print(f"  SFT  : {sft_result['generation_seconds']} s"
              f" | {sft_result['tokens_per_second']} tok/s")
        print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = args.output_dir / f"results_{timestamp}.jsonl"
    summary_path = args.output_dir / f"summary_{timestamp}.json"

    with result_path.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    def average(key, model_key):
        values = [
            item[model_key][key]
            for item in results
        ]
        return round(sum(values) / len(values), 3)

    summary = {
        "timestamp": timestamp,
        "model_parameters": base_model.num_parameters(),
        "context_len": cfg.model.context_len,
        "prompts": len(results),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "pretrained": {
            "avg_tokens_per_second": average("tokens_per_second", "pretrained"),
            "avg_generated_tokens": average("generated_tokens", "pretrained"),
            "avg_generation_seconds": average("generation_seconds", "pretrained"),
        },
        "sft": {
            "avg_tokens_per_second": average("tokens_per_second", "sft"),
            "avg_generated_tokens": average("generated_tokens", "sft"),
            "avg_generation_seconds": average("generation_seconds", "sft"),
        },
        "results_file": str(result_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== RÉSULTATS =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRéponses : {result_path}")
    print(f"Résumé   : {summary_path}")


if __name__ == "__main__":
    main()
