
#!/usr/bin/env python3
"""
Téléchargement et normalisation de datasets SFT pour TinyLLM.

Sources :
- angeluriot/french_instruct
- HuggingFaceH4/ultrachat_200k (optionnel)

Sortie :
- general_fr.jsonl
- coding_fr.jsonl
- general_en.jsonl (optionnel)
- download_report.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

from datasets import load_dataset


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "").strip()


def normalize_messages(messages):
    """Nettoie les rôles et fusionne les messages consécutifs identiques."""
    normalized = []

    for item in messages:
        role = str(item.get("role", "")).lower().strip()
        content = clean_text(item.get("content", item.get("text", "")))

        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n\n" + content
        else:
            normalized.append({
                "role": role,
                "content": content
            })

    # Le pipeline TinyLLM attend une conversation qui commence
    # par un utilisateur et se termine par une réponse assistant.
    non_system = [m for m in normalized if m["role"] != "system"]

    if not non_system:
        return None

    if non_system[0]["role"] != "user":
        return None

    if non_system[-1]["role"] != "assistant":
        return None

    if not any(m["role"] == "assistant" for m in non_system):
        return None

    # Vérifie qu'il n'y a pas deux messages identiques consécutifs
    # après normalisation.
    return normalized


def conversation_hash(messages):
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_usable(messages, min_chars=20, max_chars=12000):
    if not messages:
        return False

    total_chars = sum(len(m["content"]) for m in messages)
    assistant_chars = sum(
        len(m["content"])
        for m in messages
        if m["role"] == "assistant"
    )

    if total_chars < min_chars:
        return False

    if total_chars > max_chars:
        return False

    if assistant_chars < 10:
        return False

    return True


def french_record_to_messages(record):
    messages = []

    context = clean_text(record.get("context", ""))
    if context:
        messages.append({
            "role": "system",
            "content": context
        })

    conversation = record.get("conversation", [])

    if not isinstance(conversation, list):
        return None

    for item in conversation:
        if not isinstance(item, dict):
            continue

        messages.append({
            "role": item.get("role", ""),
            "content": item.get("text", "")
        })

    return normalize_messages(messages)


def ultrachat_record_to_messages(record):
    raw_messages = record.get("messages", [])

    if not isinstance(raw_messages, list):
        return None

    return normalize_messages(raw_messages)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as file:
        for messages in records:
            file.write(json.dumps(
                {"messages": messages},
                ensure_ascii=False
            ) + "\n")

    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/sft/downloaded/raw"
    )

    parser.add_argument(
        "--max-general-fr",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--max-coding-fr",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--include-ultrachat",
        action="store_true"
    )

    parser.add_argument(
        "--ultrachat-limit",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()

    outputs = {
        "general_fr": [],
        "coding_fr": [],
        "general_en": []
    }

    stats = Counter()

    print("\n=== TINYLLM SFT DATA DOWNLOADER ===\n")

    # --------------------------------------------------
    # 1. FRENCH INSTRUCT
    # --------------------------------------------------

    print("[1/2] Chargement de French Instruct...")

    dataset = load_dataset(
        "angeluriot/french_instruct",
        split="train"
    )

    print(f"Dataset chargé : {len(dataset):,} exemples")

    for index, record in enumerate(dataset):
        stats["french_examined"] += 1

        messages = french_record_to_messages(record)

        if not is_usable(
            messages,
            max_chars=args.max_chars
        ):
            stats["french_rejected"] += 1
            continue

        digest = conversation_hash(messages)

        if digest in seen:
            stats["duplicates"] += 1
            continue

        seen.add(digest)

        is_code = bool(record.get("code", False))

        if is_code:
            category = "coding_fr"
            limit = args.max_coding_fr
        else:
            category = "general_fr"
            limit = args.max_general_fr

        if len(outputs[category]) >= limit:
            stats[f"{category}_limit_reached"] += 1
            continue

        outputs[category].append(messages)
        stats[f"{category}_accepted"] += 1

        if (index + 1) % 25000 == 0:
            print(
                f"  Progression : {index + 1:,}/{len(dataset):,}"
            )

    print("\nFrench Instruct terminé.")
    print(f"  Général : {len(outputs['general_fr']):,}")
    print(f"  Code    : {len(outputs['coding_fr']):,}")

    # --------------------------------------------------
    # 2. ULTRACHAT OPTIONNEL
    # --------------------------------------------------

    if args.include_ultrachat:
        print("\n[2/2] Chargement d'UltraChat 200k...")

        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True
        )

        for index, record in enumerate(dataset):
            if len(outputs["general_en"]) >= args.ultrachat_limit:
                break

            stats["ultrachat_examined"] += 1

            messages = ultrachat_record_to_messages(record)

            if not is_usable(
                messages,
                max_chars=args.max_chars
            ):
                stats["ultrachat_rejected"] += 1
                continue

            digest = conversation_hash(messages)

            if digest in seen:
                stats["duplicates"] += 1
                continue

            seen.add(digest)
            outputs["general_en"].append(messages)
            stats["general_en_accepted"] += 1

            if (index + 1) % 1000 == 0:
                print(
                    f"  Progression : {index + 1:,} "
                    f"| conservés : {len(outputs['general_en']):,}"
                )

    # --------------------------------------------------
    # 3. EXPORT
    # --------------------------------------------------

    print("\nExport des fichiers...")

    for category, records in outputs.items():
        if not records:
            continue

        filename = f"{category}.jsonl"
        path = output_dir / filename

        write_jsonl(path, records)

        print(f"  {path} : {len(records):,} conversations")

    report = {
        "sources": [
            "angeluriot/french_instruct"
        ] + (
            ["HuggingFaceH4/ultrachat_200k"]
            if args.include_ultrachat else []
        ),
        "limits": {
            "general_fr": args.max_general_fr,
            "coding_fr": args.max_coding_fr,
            "general_en": args.ultrachat_limit
        },
        "counts": dict(stats),
        "exported": {
            key: len(value)
            for key, value in outputs.items()
        }
    }

    report_path = output_dir / "download_report.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== RAPPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nTéléchargement et préparation terminés.")


if __name__ == "__main__":
    main()

