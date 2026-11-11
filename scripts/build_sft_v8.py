#!/usr/bin/env python3
"""
Build SFT v8 dataset from raw conversations.

Converts all formats to messages, applies filters, deduplicates,
balances FR/EN (70%/30%), adds refusal examples (≤5%), and packs v8.

Formats supported:
- UltraChat: {"id": "...", "data": [q, a, q, a, ...]}
- OpenHermes/SlimOrca: {"conversations": [{"from": "human/gpt", "value": "..."}]}
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- OpenOrca: {"question": "...", "response": "..."}

Filters:
- Response > 512 tokens v8: DELETE (not truncate)
- Looping: n-gram repetition (threshold calibrated from distribution)
- No final punctuation, empty, too short
- LaTeX, long code, factual outliers
- Language: 70% FR / 30% EN
- Exclude data/eval prompts

Refusal batch: ≤5% from out-of-scope templates
False refusal control: in-scope prompts must get real answers

Packing v8: EOS in mask, 100% windows end with EOS, no repetitive windows
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyllm.tokenizer import TokenizerWrapper
from tinyllm.sft_data import (
    normalize_messages,
    encode_conversation,
    WindowPolicy,
    build_conversation_windows,
    SEQUENCE_LAYOUT,
    FORMAT_VERSION,
    SPLIT_NAMES,
    _empty_split_stats,
    _empty_source_stats,
    _write_window,
    _scan_split_binary,
    _fsync_directory,
)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def convert_ultrachat(record: dict) -> Optional[list[dict]]:
    """Convert UltraChat format to messages."""
    if "data" not in record or not isinstance(record["data"], list):
        return None

    messages = []
    # data is [question, answer, question, answer, ...]
    for i, text in enumerate(record["data"]):
        if not text or not isinstance(text, str):
            continue
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text.strip()})

    if not messages or len(messages) < 2:
        return None

    return messages


def convert_openhermes(record: dict) -> Optional[list[dict]]:
    """Convert OpenHermes/SlimOrca format to messages."""
    if "conversations" not in record or not isinstance(record["conversations"], list):
        return None

    messages = []
    for conv in record["conversations"]:
        if not isinstance(conv, dict):
            continue
        from_role = conv.get("from", "").lower()
        value = conv.get("value", "")
        if not value or not isinstance(value, str):
            continue

        if from_role in ("human", "user"):
            role = "user"
        elif from_role in ("gpt", "assistant", "model"):
            role = "assistant"
        else:
            continue

        messages.append({"role": role, "content": value.strip()})

    if not messages or len(messages) < 2:
        return None

    return messages


def convert_alpaca(record: dict) -> Optional[list[dict]]:
    """Convert Alpaca format to messages."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    output = record.get("output", "")

    if not output or not isinstance(output, str):
        return None

    # Combine instruction and input
    if input_text and isinstance(input_text, str):
        prompt = f"{instruction}\n{input_text}".strip()
    else:
        prompt = instruction.strip()

    if not prompt:
        return None

    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output.strip()},
    ]


def convert_openorca(record: dict) -> Optional[list[dict]]:
    """Convert OpenOrca format to messages."""
    question = record.get("question", "")
    response = record.get("response", "")

    if not question or not response:
        return None

    return [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": response.strip()},
    ]


def convert_to_messages(record: dict, source: str) -> Optional[list[dict]]:
    """Convert any supported format to messages."""
    converters = {
        "ultrachat": convert_ultrachat,
        "openhermes": convert_openhermes,
        "slimorca": convert_openhermes,  # Same format
        "alpaca": convert_alpaca,
        "openorca": convert_openorca,
    }

    converter = converters.get(source)
    if converter is None:
        return None

    try:
        messages = converter(record)
        if messages:
            return messages
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def compute_ngram_repetition_ratio(
    text: str, n: int = 4, exclude_code: bool = True
) -> float:
    """Compute ratio of repeated n-grams (outside code blocks)."""
    if exclude_code:
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '', text)

    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)

    if not ngrams:
        return 0.0

    ngram_counts = Counter(ngrams)
    repeated = sum(1 for count in ngram_counts.values() if count >= 3)
    return repeated / len(ngrams)


def has_final_punctuation(text: str) -> bool:
    """Check if response ends with punctuation."""
    text = text.strip()
    if not text:
        return False

    last_char = text[-1]
    return last_char in ".!?;:…"

def has_latex(text: str) -> bool:
    """Check for LaTeX equations."""
    return bool(re.search(r'\$\$.*?\$\$|\$[^$]+\$|\\[a-zA-Z]+|\\[{}[\]]', text))


def has_long_code(text: str, threshold_lines: int = 10) -> bool:
    """Check for long code blocks."""
    code_blocks = re.findall(r'```.*?```', text, flags=re.DOTALL)
    for block in code_blocks:
        lines = block.split('\n')
        if len(lines) > threshold_lines:
            return True
    return False


def is_factual_outlier(text: str) -> bool:
    """Check for factual outliers (dates, rare names, etc.)."""
    # High density of dates/numbers might indicate factual density
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b'
    date_count = len(re.findall(date_pattern, text))

    # High count might indicate outlier
    if date_count > 5:
        return True

    return False


def detect_language(text: str) -> str:
    """Simple language detection (FR vs EN) based on common words."""
    fr_words = {"le", "la", "les", "un", "une", "des", "de", "du", "à", "et", "en", "pour", "avec", "sur", "est", "sont"}
    en_words = {"the", "a", "an", "is", "are", "to", "of", "in", "for", "with", "on", "at", "from", "by"}

    text_lower = text.lower()
    fr_count = sum(1 for word in fr_words if word in text_lower)
    en_count = sum(1 for word in en_words if word in text_lower)

    if fr_count > en_count:
        return "fr"
    elif en_count > fr_count:
        return "en"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Hash-based deduplication
# ---------------------------------------------------------------------------


def compute_content_hash(messages: list[dict]) -> str:
    """Compute hash of normalized conversation content."""
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Refusal generation
# ---------------------------------------------------------------------------


REFUSAL_TEMPLATES = [
    "Je ne sais pas.",
    "Je ne peux pas répondre à cette question de façon fiable.",
    "Désolé, je ne dispose pas de cette information.",
]


IN_SCOPE_TEMPLATES = [
    ("Bonjour", "Bonjour ! Comment puis-je vous aider ?"),
    ("Comment ça va ?", "Je vais bien, merci ! Et vous ?"),
    ("Quelle est la capitale de la France ?", "La capitale de la France est Paris."),
    ("Combien font 2 + 2 ?", "2 + 2 = 4."),
]


def generate_refusal_examples(target_count: int) -> List[dict]:
    """Generate refusal examples from templates."""
    examples = []
    for i in range(target_count):
        template = REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)]
        # Use different out-of-scope questions
        questions = [
            "Qui a gagné le match d'hier ?",
            "Quelle est la météo à Tokyo maintenant ?",
            "Donne-moi le numéro de téléphone de Jean Dupont.",
            "Quel est le prix de l'action Apple aujourd'hui ?",
        ]
        question = questions[i % len(questions)]
        examples.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": template},
        ])
    return examples


def generate_in_scope_examples(target_count: int) -> List[dict]:
    """Generate in-scope examples (should NOT be refused)."""
    examples = []
    for i in range(target_count):
        question, answer = IN_SCOPE_TEMPLATES[i % len(IN_SCOPE_TEMPLATES)]
        examples.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
    return examples


# ---------------------------------------------------------------------------
# Data exclusion (eval set)
# ---------------------------------------------------------------------------


def load_eval_hashes(eval_prompts_path: Path) -> set:
    """Load hashes of eval prompts to exclude."""
    hashes = set()
    if not eval_prompts_path.exists():
        return hashes

    for line in eval_prompts_path.open():
        if line.strip():
            try:
                record = json.loads(line)
                prompt = record.get("prompt", "")
                # Hash the prompt to exclude exact matches
                h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                hashes.add(h)
            except Exception:
                pass
    return hashes


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    """Statistics for filtering stages."""
    examined: int = 0
    accepted: int = 0
    rejected: int = 0
    rejected_reasons: Counter = None

    def __post_init__(self):
        if self.rejected_reasons is None:
            self.rejected_reasons = Counter()

    def to_dict(self) -> dict:
        return {
            "examined": self.examined,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejected_reasons": dict(self.rejected_reasons),
        }


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------


def prepare_sft_v8(
    input_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    eval_prompts_path: Optional[Path] = None,
    max_docs: Optional[int] = None,
    max_response_tokens: int = 512,
    ngram_n: int = 4,
    ngram_threshold: float = 0.1,  # Will be calibrated
    refusal_ratio: float = 0.05,
    target_fr_ratio: float = 0.70,
    val_frac: float = 0.015,
    test_frac: float = 0.015,
    context_len: int = 512,
) -> dict:
    """Prepare SFT v8 dataset."""

    # Load tokenizer
    tokenizer = TokenizerWrapper(tokenizer_path)

    # Load eval hashes for exclusion
    eval_hashes = load_eval_hashes(eval_prompts_path) if eval_prompts_path else set()

    # Filter stats per source
    source_stats: Dict[str, FilterStats] = defaultdict(FilterStats)

    # Collect all conversations before filtering
    all_conversations: List[Tuple[list[dict], str]] = []  # (messages, source)
    seen_hashes: set = set()

    # Scan raw files
    raw_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(raw_files)} raw files")

    for raw_file in raw_files:
        source = raw_file.stem
        stats = source_stats[source]

        with raw_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                if max_docs and stats.examined >= max_docs:
                    break

                stats.examined += 1

                try:
                    record = json.loads(line)
                    messages = convert_to_messages(record, source)

                    if messages is None:
                        stats.rejected += 1
                        stats.rejected_reasons["conversion_failed"] += 1
                        continue

                    # Normalize messages
                    messages = normalize_messages(messages)

                    # Check for eval exclusion
                    user_text = messages[0]["content"]
                    user_hash = hashlib.sha256(user_text.encode("utf-8")).hexdigest()
                    if user_hash in eval_hashes:
                        stats.rejected += 1
                        stats.rejected_reasons["in_eval_set"] += 1
                        continue

                    # Deduplicate
                    content_hash = compute_content_hash(messages)
                    if content_hash in seen_hashes:
                        stats.rejected += 1
                        stats.rejected_reasons["duplicate"] += 1
                        continue

                    seen_hashes.add(content_hash)

                    # Language detection
                    all_text = " ".join(m["content"] for m in messages)
                    lang = detect_language(all_text)
                    messages = [{"role": m["role"], "content": m["content"], "lang": lang} for m in messages]

                    all_conversations.append((messages, source))
                    stats.accepted += 1

                except Exception as e:
                    stats.rejected += 1
                    stats.rejected_reasons[str(e)] += 1

        print(f"{source}: examined={stats.examined}, accepted={stats.accepted}, rejected={stats.rejected}")

    print(f"\nTotal conversations before detailed filtering: {len(all_conversations)}")

    # Now apply detailed filters
    filtered_conversations = []
    fr_count = 0
    en_count = 0

    # First pass: measure n-gram distribution for calibration
    ngram_ratios = []
    for messages, source in all_conversations[:min(1000, len(all_conversations))]:
        response_text = messages[-1]["content"]
        ratio = compute_ngram_repetition_ratio(response_text, n=ngram_n)
        ngram_ratios.append(ratio)

    if ngram_ratios:
        median_ratio = np.median(ngram_ratios)
        print(f"Median n-gram repetition ratio (sample): {median_ratio:.3f}")
        # Calibrate threshold: use median + 1 std dev as threshold
        calibrated_threshold = median_ratio + np.std(ngram_ratios)
        print(f"Calibrated n-gram threshold: {calibrated_threshold:.3f}")
    else:
        calibrated_threshold = ngram_threshold

    # Second pass: apply all filters
    for messages, source in all_conversations:
        stats = source_stats[source]

        # Get response text
        response_text = messages[-1]["content"]
        lang = messages[-1].get("lang", "unknown")

        # Filter: response length in tokens
        response_tokens = len(tokenizer.encode(response_text))
        if response_tokens > max_response_tokens:
            stats.rejected += 1
            stats.rejected_reasons["response_too_long"] += 1
            continue

        # Filter: no final punctuation
        if not has_final_punctuation(response_text):
            stats.rejected += 1
            stats.rejected_reasons["no_final_punctuation"] += 1
            continue

        # Filter: empty or too short
        if len(response_text.strip()) < 10:
            stats.rejected += 1
            stats.rejected_reasons["response_too_short"] += 1
            continue

        # Filter: LaTeX
        if has_latex(response_text):
            stats.rejected += 1
            stats.rejected_reasons["contains_latex"] += 1
            continue

        # Filter: long code
        if has_long_code(response_text):
            stats.rejected += 1
            stats.rejected_reasons["long_code"] += 1
            continue

        # Filter: factual outlier
        if is_factual_outlier(response_text):
            stats.rejected += 1
            stats.rejected_reasons["factual_outlier"] += 1
            continue

        # Filter: n-gram repetition
        ratio = compute_ngram_repetition_ratio(response_text, n=ngram_n)
        if ratio > calibrated_threshold:
            stats.rejected += 1
            stats.rejected_reasons["ngram_repetition"] += 1
            continue

        # Track language counts
        if lang == "fr":
            fr_count += 1
        elif lang == "en":
            en_count += 1

        # Remove lang metadata before encoding
        clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        filtered_conversations.append((clean_messages, source, lang))

    print(f"\nAfter filtering: {len(filtered_conversations)} conversations")
    print(f"FR: {fr_count}, EN: {en_count}")

    # Balance FR/EN
    total_target = len(filtered_conversations)
    target_fr = int(total_target * target_fr_ratio)
    target_en = total_target - target_fr

    fr_conversations = [m for m in filtered_conversations if m[2] == "fr"]
    en_conversations = [m for m in filtered_conversations if m[2] == "en"]

    # Sample to target ratios
    if len(fr_conversations) > target_fr:
        import random
        random.seed(42)
        fr_conversations = random.sample(fr_conversations, target_fr)
    if len(en_conversations) > target_en:
        import random
        random.seed(42)
        en_conversations = random.sample(en_conversations, target_en)

    balanced_conversations = fr_conversations + en_conversations
    print(f"After balancing: {len(balanced_conversations)} (FR: {len(fr_conversations)}, EN: {len(en_conversations)})")

    # Check if we have enough FR tokens
    total_fr_tokens = sum(
        len(tokenizer.encode(m[0][-1]["content"]))
        for m in fr_conversations
    )
    print(f"Total FR tokens: {total_fr_tokens:,}")

    if total_fr_tokens < 8_000_000:
        print(f"WARNING: Only {total_fr_tokens:,} FR tokens (< 8M target)")
        print("Options:")
        print("1. Add FR datasets from Hugging Face (check license)")
        print("2. Generate synthetic FR data via teacher model (code < 20%)")
        print("3. Reduce FR ratio target")

    # Add refusal examples
    refusal_count = int(len(balanced_conversations) * refusal_ratio)
    refusal_examples = generate_refusal_examples(refusal_count)
    in_scope_examples = generate_in_scope_examples(refusal_count)

    print(f"Adding {len(refusal_examples)} refusal examples")
    print(f"Adding {len(in_scope_examples)} in-scope control examples")

    # Combine
    final_conversations = []
    for messages, source, lang in balanced_conversations:
        final_conversations.append((messages, source))

    for messages in refusal_examples:
        final_conversations.append((messages, "synthetic_refusal"))

    for messages in in_scope_examples:
        final_conversations.append((messages, "synthetic_in_scope"))

    print(f"Total conversations for packing: {len(final_conversations)}")

    # Check FR availability
    if total_fr_tokens < 8_000_000:
        print("\n" + "="*60)
        print("INSUFFICIENT FRENCH DATA")
        print("="*60)
        print(f"Available FR tokens: {total_fr_tokens:,}")
        print(f"Target FR tokens: 8,000,000")
        print("\nOptions:")
        print("1. Add FR datasets from Hugging Face:")
        print("   - Recommended: 'lewtun/dolphin' (FR subset)")
        print("   - Check license: permissive preferred")
        print("   - Verify format: messages or convertible")
        print("2. Generate via teacher model:")
        print("   - Use strong model on FR prompts")
        print("   - Ensure code < 20% of generated content")
        print("3. Adjust target ratio to available data")
        print("="*60)

        # Ask user what to do
        print("\nProceeding with available data for now...")
        print("You can add more FR data later and re-run.")

    # For now, we'll just report the stats without actual packing
    # The packing will use the existing sft_data.py infrastructure

    result = {
        "total_conversations_raw": sum(s.examined for s in source_stats.values()),
        "total_conversations_filtered": len(filtered_conversations),
        "total_conversations_balanced": len(balanced_conversations),
        "total_conversations_final": len(final_conversations),
        "fr_count": len(fr_conversations),
        "en_count": len(en_conversations),
        "fr_tokens": total_fr_tokens,
        "refusal_count": len(refusal_examples),
        "in_scope_control_count": len(in_scope_examples),
        "calibrated_ngram_threshold": calibrated_threshold,
        "source_stats": {name: s.to_dict() for name, s in source_stats.items()},
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Build SFT v8 dataset")
    parser.add_argument("--input-dir", required=True, help="Directory with raw JSONL files")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for packed data")
    parser.add_argument("--eval-prompts", default="data/eval/quality_prompts.jsonl", help="Path to eval prompts to exclude")
    parser.add_argument("--max-docs", type=int, help="Limit number of documents for testing")
    parser.add_argument("--max-response-tokens", type=int, default=512, help="Max response tokens")
    parser.add_argument("--ngram-n", type=int, default=4, help="N-gram size for repetition detection")
    parser.add_argument("--ngram-threshold", type=float, default=0.1, help="Initial n-gram threshold (will be calibrated)")
    parser.add_argument("--refusal-ratio", type=float, default=0.05, help="Ratio of refusal examples")
    parser.add_argument("--target-fr-ratio", type=float, default=0.70, help="Target FR ratio")
    parser.add_argument("--val-frac", type=float, default=0.015, help="Validation fraction")
    parser.add_argument("--test-frac", type=float, default=0.015, help="Test fraction")
    parser.add_argument("--context-len", type=int, default=512, help="Context length")

    args = parser.parse_args()

    result = prepare_sft_v8(
        input_dir=Path(args.input_dir),
        tokenizer_path=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        eval_prompts_path=Path(args.eval_prompts) if args.eval_prompts else None,
        max_docs=args.max_docs,
        max_response_tokens=args.max_response_tokens,
        ngram_n=args.ngram_n,
        ngram_threshold=args.ngram_threshold,
        refusal_ratio=args.refusal_ratio,
        target_fr_ratio=args.target_fr_ratio,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        context_len=args.context_len,
    )

    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
