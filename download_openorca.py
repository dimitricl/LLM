from datasets import load_dataset
import json
from pathlib import Path

# Télécharger OpenOrca
dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

output_path = Path("data/sft/downloaded/raw/openorca.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

count = 0
max_examples = 100000

with output_path.open("w", encoding="utf-8") as f:
    for record in dataset:
        if count >= max_examples:
            break

        # Format OpenOrca : system_prompt, question, response
        messages = []

        system = record.get("system_prompt", "")
        if system:
            messages.append({"role": "system", "content": system})

        question = record.get("question", "")
        if question:
            messages.append({"role": "user", "content": question})

        response = record.get("response", "")
        if response:
            messages.append({"role": "assistant", "content": response})

        if messages:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

        if count % 1000 == 0:
            print(f"Téléchargés : {count}/{max_examples}")

print(f"Terminé : {count} conversations")
