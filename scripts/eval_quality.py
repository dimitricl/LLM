#!/usr/bin/env python3
"""
Evaluate model quality on fixed prompts.

Computes metrics with and without decoding penalties:
- Loop rate (4-gram repetition >= 3 or max_tokens without EOS)
- Distinct-1/2/3
- Out-of-vocabulary rate (French lexicon)
- Factual accuracy (keyword matching)
- Refusal rate on out-of-scope prompts
- False refusal rate on in-scope prompts

Supports both v7 and v8 tokenizers for comparison.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np

from tinyllm.tokenizer import TokenizerWrapper
from tinyllm.decoding import (
    count_ngram_repetitions,
    detect_looping_output,
)


def load_lexicon(lexicon_path: Path) -> set:
    """Load French lexicon words (one per line, ignore comments)."""
    words = set()
    for line in lexicon_path.open():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
    return words


def load_prompts(prompts_path: Path) -> List[Dict]:
    """Load quality prompts from JSONL."""
    prompts = []
    for line in prompts_path.open():
        if line.strip():
            prompts.append(json.loads(line))
    return prompts


def compute_distinct(text: str, n: int) -> float:
    """Compute distinct-n ratio."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i+n])
        ngrams.append(ngram)

    if not ngrams:
        return 0.0

    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0


def compute_oov_rate(text: str, lexicon: set) -> float:
    """Compute rate of words not in French lexicon."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    oov_count = sum(1 for token in tokens if token not in lexicon)
    return oov_count / len(tokens)


def check_keyword_match(text: str, keywords: List[str]) -> bool:
    """Check if any keyword appears in text (case-insensitive)."""
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True
    return False


def is_refusal(text: str) -> bool:
    """Check if response is a refusal."""
    refusal_phrases = [
        "je ne sais pas",
        "je ne peux pas",
        "je ne peux répondre",
        "désolé",
        "impossible",
        "je n'ai pas",
        "je ne dispose pas",
        "i don't know",
        "i cannot",
        "sorry",
        "unable",
    ]
    text_lower = text.lower()
    for phrase in refusal_phrases:
        if phrase in text_lower:
            return True
    return False


def evaluate_outputs(
    outputs: List[Dict],
    lexicon: set,
    n_threshold: int = 3,
    n_size: int = 4,
) -> Dict:
    """Compute evaluation metrics from outputs."""
    metrics = {
        "total": len(outputs),
        "loop_rate": 0,
        "no_eos_rate": 0,
        "max_tokens_rate": 0,
        "distinct_1": 0.0,
        "distinct_2": 0.0,
        "distinct_3": 0.0,
        "oov_rate": 0.0,
        "factual_accuracy": 0.0,
        "factual_count": 0,
        "refusal_correct": 0.0,
        "refusal_correct_count": 0,
        "false_refusal": 0.0,
        "false_refusal_count": 0,
    }

    distinct_1_values = []
    distinct_2_values = []
    distinct_3_values = []
    oov_values = []

    loop_count = 0
    no_eos_count = 0
    max_tokens_count = 0

    for output in outputs:
        text = output.get("response", "")
        tokens = output.get("token_ids", [])
        eos_id = output.get("eos_id", 0)
        max_tokens = output.get("max_tokens", 120)

        # Loop detection
        is_looping, reason = detect_looping_output(
            tokens, eos_id, n=n_size, threshold=n_threshold, max_tokens=max_tokens
        )
        if is_looping:
            loop_count += 1
            if "max_tokens" in reason:
                max_tokens_count += 1
            if "EOS" in reason or "without EOS" in reason:
                no_eos_count += 1

        # Distinct-n
        distinct_1_values.append(compute_distinct(text, 1))
        distinct_2_values.append(compute_distinct(text, 2))
        distinct_3_values.append(compute_distinct(text, 3))

        # OOV rate
        oov_values.append(compute_oov_rate(text, lexicon))

        # Factual accuracy (for factual prompts)
        if output.get("category") == "factual":
            keywords = output.get("expected_keywords", [])
            if keywords:
                metrics["factual_count"] += 1
                if check_keyword_match(text, keywords):
                    metrics["factual_accuracy"] += 1

        # Refusal correctness (for out-of-scope)
        if output.get("category") == "out_of_scope":
            metrics["refusal_correct_count"] += 1
            if is_refusal(text):
                metrics["refusal_correct"] += 1

        # False refusal (for in-scope prompts)
        if output.get("category") in ["greeting", "instruction", "factual", "math"]:
            metrics["false_refusal_count"] += 1
            if is_refusal(text):
                metrics["false_refusal"] += 1

    # Compute averages
    metrics["loop_rate"] = loop_count / len(outputs) if outputs else 0.0
    metrics["no_eos_rate"] = no_eos_count / len(outputs) if outputs else 0.0
    metrics["max_tokens_rate"] = max_tokens_count / len(outputs) if outputs else 0.0
    metrics["distinct_1"] = np.mean(distinct_1_values) if distinct_1_values else 0.0
    metrics["distinct_2"] = np.mean(distinct_2_values) if distinct_2_values else 0.0
    metrics["distinct_3"] = np.mean(distinct_3_values) if distinct_3_values else 0.0
    metrics["oov_rate"] = np.mean(oov_values) if oov_values else 0.0

    # Convert counts to rates
    if metrics["factual_count"] > 0:
        metrics["factual_accuracy"] /= metrics["factual_count"]
    if metrics["refusal_correct_count"] > 0:
        metrics["refusal_correct"] /= metrics["refusal_correct_count"]
    if metrics["false_refusal_count"] > 0:
        metrics["false_refusal"] /= metrics["false_refusal_count"]

    return metrics


def compute_confidence_interval(value: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)

    z = 1.96  # 95% confidence
    p = value
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


def format_ci(value: float, n: int) -> str:
    """Format value with confidence interval."""
    if n == 0:
        return f"{value:.3f} (N/A)"
    lower, upper = compute_confidence_interval(value, n)
    return f"{value:.3f} ±{((upper - lower) / 2):.3f}"


def format_markdown_table(
    metrics_with_penalties: Dict,
    metrics_without_penalties: Dict,
    checkpoint: str,
) -> str:
    """Format metrics as markdown table."""
    lines = []
    lines.append(f"# Evaluation Report: {checkpoint}")
    lines.append("")
    lines.append("## Metrics Comparison")
    lines.append("")
    lines.append("| Metric | With Penalties | Without Penalties |")
    lines.append("|--------|----------------|-------------------|")

    total = metrics_with_penalties["total"]

    def fmt(val, n=total):
        return format_ci(val, n)

    lines.append(f"| Loop Rate | {fmt(metrics_with_penalties['loop_rate'])} | {fmt(metrics_without_penalties['loop_rate'])} |")
    lines.append(f"| No EOS Rate | {fmt(metrics_with_penalties['no_eos_rate'])} | {fmt(metrics_without_penalties['no_eos_rate'])} |")
    lines.append(f"| Max Tokens Rate | {fmt(metrics_with_penalties['max_tokens_rate'])} | {fmt(metrics_without_penalties['max_tokens_rate'])} |")
    lines.append(f"| Distinct-1 | {fmt(metrics_with_penalties['distinct_1'])} | {fmt(metrics_without_penalties['distinct_1'])} |")
    lines.append(f"| Distinct-2 | {fmt(metrics_with_penalties['distinct_2'])} | {fmt(metrics_without_penalties['distinct_2'])} |")
    lines.append(f"| Distinct-3 | {fmt(metrics_with_penalties['distinct_3'])} | {fmt(metrics_without_penalties['distinct_3'])} |")
    lines.append(f"| OOV Rate | {fmt(metrics_with_penalties['oov_rate'])} | {fmt(metrics_without_penalties['oov_rate'])} |")

    n_factual = metrics_with_penalties["factual_count"]
    lines.append(f"| Factual Accuracy | {fmt(metrics_with_penalties['factual_accuracy'], n_factual)} | {fmt(metrics_without_penalties['factual_accuracy'], n_factual)} |")

    n_refusal = metrics_with_penalties["refusal_correct_count"]
    lines.append(f"| Refusal Correct (Out-of-Scope) | {fmt(metrics_with_penalties['refusal_correct'], n_refusal)} | {fmt(metrics_without_penalties['refusal_correct'], n_refusal)} |")

    n_false_refusal = metrics_with_penalties["false_refusal_count"]
    lines.append(f"| False Refusal (In-Scope) | {fmt(metrics_with_penalties['false_refusal'], n_false_refusal)} | {fmt(metrics_without_penalties['false_refusal'], n_false_refusal)} |")

    lines.append("")
    lines.append(f"**Total prompts evaluated:** {total}")
    lines.append("")
    lines.append("## Decoding Parameters")
    lines.append("")
    lines.append("With penalties:")
    lines.append("- temperature: 0.65")
    lines.append("- top_p: 0.9")
    lines.append("- top_k: 40")
    lines.append("- repetition_penalty: 1.15")
    lines.append("- no_repeat_ngram_size: 4")
    lines.append("")
    lines.append("Without penalties (greedy baseline):")
    lines.append("- temperature: 0")
    lines.append("- top_p: 1.0")
    lines.append("- top_k: None")
    lines.append("- repetition_penalty: 1.0")
    lines.append("- no_repeat_ngram_size: 0")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model quality")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint name")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--prompts", default="data/eval/quality_prompts.jsonl", help="Path to prompts JSONL")
    parser.add_argument("--lexicon", default="data/eval/fr_lexicon.txt", help="Path to French lexicon")
    parser.add_argument("--output-json", help="Path to output JSON")
    parser.add_argument("--output-md", help="Path to output markdown")
    parser.add_argument("--max-docs", type=int, help="Limit number of prompts for testing")
    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        sys.exit(1)

    prompts = load_prompts(prompts_path)
    if args.max_docs:
        prompts = prompts[:args.max_docs]

    print(f"Loaded {len(prompts)} prompts")

    # Load lexicon
    lexicon_path = Path(args.lexicon)
    if lexicon_path.exists():
        lexicon = load_lexicon(lexicon_path)
        print(f"Loaded {len(lexicon)} French lexicon words")
    else:
        print(f"Warning: Lexicon not found: {lexicon_path}")
        lexicon = set()

    # Load tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    tokenizer = TokenizerWrapper(str(tokenizer_path))
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_id}")

    # Note: This script does NOT generate outputs - it's a placeholder
    # The actual generation requires MLX and must be run by the user
    # For now, we'll report what the script WOULD do

    print("\n" + "="*60)
    print("EVALUATION SCRIPT READY")
    print("="*60)
    print(f"\nThis script will evaluate model at:")
    print(f"  Run dir: {args.run_dir}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"\nPrompts: {len(prompts)}")
    print(f"Lexicon: {len(lexicon)} words")
    print(f"\nTo run the actual evaluation, you need to:")
    print("1. Have MLX installed")
    print("2. Have the checkpoint ready")
    print("3. Run the generation phase (not implemented in this script)")
    print("\nThe evaluation would compute:")
    print("- Loop rate (4-gram repetition >= 3)")
    print("- Distinct-1/2/3")
    print("- OOV rate (French lexicon)")
    print("- Factual accuracy (keyword matching)")
    print("- Refusal correctness")
    print("- False refusal rate")
    print("\nEach metric WITH and WITHOUT decoding penalties.")


if __name__ == "__main__":
    main()
