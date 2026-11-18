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


def convert_messages(record: dict) -> Optional[list[dict]]:
    """Convert an already normalised ``{"messages": [...]}`` record."""
    raw_messages = record.get("messages")

    if not isinstance(raw_messages, list):
        return None

    messages = []

    for message in raw_messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            continue

        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "model"):
            role = "assistant"
        elif role != "system":
            continue

        messages.append({"role": role, "content": content.strip()})

    if len(messages) < 2:
        return None

    return messages


# Filename hints are only used as a fallback: most raw files carry their
# format in the record itself, and several sources (general_fr/en,
# coding_fr, teacher_responses) are already in ``messages`` form.
_CONVERTERS = {
    "ultrachat": convert_ultrachat,
    "openhermes": convert_openhermes,
    "slimorca": convert_openhermes,  # Same format
    "alpaca": convert_alpaca,
    "openorca": convert_openorca,
    "general_fr": convert_messages,
    "general_en": convert_messages,
    "coding_fr": convert_messages,
    "teacher_responses": convert_messages,
    "combined_with_teacher": convert_messages,
    "combined_v7": convert_messages,
}


def _detect_converter(record: dict):
    """Pick a converter from the record shape, not from the filename."""
    if isinstance(record.get("messages"), list):
        return convert_messages
    if isinstance(record.get("conversations"), list):
        return convert_openhermes
    if isinstance(record.get("data"), list):
        return convert_ultrachat
    if "instruction" in record and "output" in record:
        return convert_alpaca
    if "question" in record and "response" in record:
        return convert_openorca
    return None


def convert_to_messages(record: dict, source: str) -> Optional[list[dict]]:
    """Convert any supported record shape to normalised messages."""
    converter = _detect_converter(record) or _CONVERTERS.get(source)

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


_FR_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "est",
    "sont", "pour", "avec", "sur", "dans", "que", "qui", "ne", "pas",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "ce",
    "cette", "ces", "au", "aux", "en", "se", "sa", "son", "ses", "mais",
    "ou", "où", "donc", "car", "être", "avoir", "fait", "plus", "comme",
}

_EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "for", "with", "on", "at", "from", "by", "and", "that", "this",
    "it", "you", "we", "they", "be", "as", "or", "not", "have", "has",
    "do", "does", "can", "will", "would", "should", "there", "their",
}

_WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ']+")


def detect_language(text: str) -> str:
    """Detect FR vs EN with word-boundary stopword counts.

    The previous substring check (``"a" in text``) counted almost every
    text as English; this uses whole words only.
    """
    words = [word.lower() for word in _WORD_RE.findall(text)]

    if not words:
        return "unknown"

    fr_count = sum(1 for word in words if word in _FR_STOPWORDS)
    en_count = sum(1 for word in words if word in _EN_STOPWORDS)

    # Accented characters are a strong French signal.
    if sum(1 for char in text if char in "àâäéèêëîïôöùûüçœ") >= 3:
        fr_count += 2

    if fr_count == en_count:
        return "unknown"
    return "fr" if fr_count > en_count else "en"


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
    sources: Optional[List[str]] = None,
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
    raw_files = sorted(input_dir.glob("*.jsonl"))

    if sources:
        wanted = {name.strip() for name in sources if name.strip()}
        unknown = wanted - {path.stem for path in raw_files}
        if unknown:
            raise FileNotFoundError(
                f"Unknown source(s): {sorted(unknown)}"
            )
        raw_files = [path for path in raw_files if path.stem in wanted]

    if not raw_files:
        raise FileNotFoundError(f"No .jsonl source files found in {input_dir}")

    print(f"Found {len(raw_files)} raw files: {[p.stem for p in raw_files]}")

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

                except Exception as e:
                    stats.rejected += 1
                    stats.rejected_reasons[str(e)] += 1

        print(f"{source}: examined={stats.examined}, kept={stats.examined - stats.rejected}")

    print(f"\nTotal conversations before detailed filtering: {len(all_conversations)}")

    # Now apply detailed filters
    filtered_conversations = []
    fr_count = 0
    en_count = 0

    # First pass: measure n-gram distribution for calibration.  Sample evenly
    # across the whole corpus instead of only the alphabetically-first source.
    sample_step = max(1, len(all_conversations) // 1000)
    ngram_ratios = []
    for messages, source in all_conversations[::sample_step]:
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
        stats.accepted += 1

    for source, stats in source_stats.items():
        print(
            f"{source}: accepted={stats.accepted}, "
            f"rejected={stats.rejected}/{stats.examined}"
        )

    print(f"\nAfter filtering: {len(filtered_conversations)} conversations")
    print(f"FR: {fr_count}, EN: {en_count}")

    # Balance FR/EN.  If one language is the limiting factor, cap the other
    # side so the requested ratio is actually reached instead of silently
    # letting the majority language dominate.
    fr_conversations = [m for m in filtered_conversations if m[2] == "fr"]
    en_conversations = [m for m in filtered_conversations if m[2] == "en"]

    import random
    random.seed(42)

    if target_fr_ratio >= 1.0:
        en_target = 0
    else:
        en_target = int(len(fr_conversations) * (1.0 - target_fr_ratio) / target_fr_ratio)

    if len(en_conversations) > en_target:
        en_conversations = random.sample(en_conversations, en_target)

    # Recompute the FR cap from the (possibly reduced) EN set so the ratio
    # is preserved when both sides have spare data.
    fr_target = (
        len(filtered_conversations)
        if target_fr_ratio <= 0.0
        else int(len(en_conversations) * target_fr_ratio / (1.0 - target_fr_ratio))
    )
    if len(fr_conversations) > fr_target:
        fr_conversations = random.sample(fr_conversations, fr_target)

    balanced_conversations = fr_conversations + en_conversations
    random.shuffle(balanced_conversations)
    print(f"After balancing: {len(balanced_conversations)} (FR: {len(fr_conversations)}, EN: {len(en_conversations)})")

    # Check how many FR tokens are actually available.
    total_fr_tokens = sum(
        len(tokenizer.encode(m[0][-1]["content"]))
        for m in fr_conversations
    )
    print(f"Total FR tokens: {total_fr_tokens:,}")
    if total_fr_tokens < 8_000_000:
        print(
            f"WARNING: only {total_fr_tokens:,} FR tokens (< 8M target); "
            "proceeding with available data."
        )

    # Add refusal examples.  Synthetic prompts must not leak into the eval set.
    refusal_count = int(len(balanced_conversations) * refusal_ratio)
    refusal_examples = [
        example
        for example in generate_refusal_examples(refusal_count)
        if hashlib.sha256(example[0]["content"].encode("utf-8")).hexdigest()
        not in eval_hashes
    ]
    in_scope_examples = [
        example
        for example in generate_in_scope_examples(refusal_count)
        if hashlib.sha256(example[0]["content"].encode("utf-8")).hexdigest()
        not in eval_hashes
    ]

    print(f"Adding {len(refusal_examples)} refusal examples")
    print(f"Adding {len(in_scope_examples)} in-scope control examples")

    # Combine
    final_conversations = []
    for messages, source, lang in balanced_conversations:
        final_conversations.append((messages, source))

    seen_final = set()
    for messages, source in final_conversations:
        seen_final.add(compute_content_hash(messages))

    for messages in refusal_examples:
        digest = compute_content_hash(messages)
        if digest not in seen_final:
            seen_final.add(digest)
            final_conversations.append((messages, "synthetic_refusal"))

    for messages in in_scope_examples:
        digest = compute_content_hash(messages)
        if digest not in seen_final:
            seen_final.add(digest)
            final_conversations.append((messages, "synthetic_in_scope"))

    print(f"Total conversations for packing: {len(final_conversations)}")


    # Write conversations to temporary JSONL for packing
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_jsonl = temp_dir / "conversations.jsonl"
    with temp_jsonl.open("w", encoding="utf-8") as f:
        for messages, source in final_conversations:
            f.write(json.dumps({"messages": messages, "source": source}, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(final_conversations)} conversations to {temp_jsonl}")

    # Now use sft_data.prepare for packing
    from tinyllm.sft_data import prepare

    policy = WindowPolicy(
        context_len=context_len,
        max_question_tokens=256,
        min_response_context=64,
        min_history_context=32,
    )

    print("\nPacking conversations...")
    metadata = prepare(
        input_paths=[temp_jsonl],
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        context_len=context_len,
        val_frac=val_frac,
        test_frac=test_frac,
        min_assistant_chars=1,
        max_question_tokens=256,
        min_response_context=64,
        min_history_context=32,
    )

    # Clean up temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

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
        "packing_metadata": metadata,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Build SFT v8 dataset")
    parser.add_argument("--input-dir", required=True, help="Directory with raw JSONL files")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for packed data")
    parser.add_argument("--eval-prompts", default="data/eval/quality_prompts.jsonl", help="Path to eval prompts to exclude")
    parser.add_argument("--max-docs", type=int, help="Limit number of documents per source for testing")
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source stems to read (default: all *.jsonl)",
    )
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
        sources=args.sources.split(",") if args.sources else None,
    )

    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
