# Commandes pour TinyLLM

## 1. Tester le modèle actuel (sft_v3_large)

### Créer le config.json pour le chat
```bash
cat > runs/sft_v3_large/config.json << 'EOF'
{
  "run_name": "sft_v3_large",
  "output_dir": "runs",
  "model": {
    "vocab_size": 16384,
    "d_model": 512,
    "n_layers": 12,
    "n_heads": 8,
    "n_kv_heads": 2,
    "ffn_hidden": 1408,
    "context_len": 512,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-05
  },
  "train": {
    "micro_batch": 4,
    "grad_accum": 8,
    "lr": 2e-05,
    "min_lr_ratio": 0.1,
    "warmup_ratio": 0.02,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "max_steps": 5000,
    "eval_interval": 100,
    "checkpoint_interval": 500,
    "log_interval": 10,
    "sample_interval": 1000,
    "max_hours": 48,
    "seed": 42,
    "dtype": "bfloat16"
  },
  "data": {
    "tokenizer_path": "data/tokenizer.json",
    "train_bin": "data/phase1/train.bin",
    "val_bin": "data/phase1/val.bin",
    "sources": {
      "tinystories": 1.0
    },
    "val_frac": 0.01,
    "test_frac": 0.01,
    "min_chars": 200,
    "max_chars": 100000,
    "test_bin": "data/phase1/test.bin"
  }
}
EOF
```

### Évaluer le modèle
```bash
.venv/bin/python -m tinyllm.sft_eval \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --sft-checkpoint runs/sft_v3_large/checkpoints/best \
  --test-tokens data/sft/packed_v3/test.tokens.bin \
  --test-mask data/sft/packed_v3/test.mask.bin
```

### Tester en chat interactif
```bash
.venv/bin/python -m tinyllm.chat \
  --run-dir runs/sft_v3_large \
  --checkpoint best
```

### Générer du texte simple
```bash
.venv/bin/python -m tinyllm.sample \
  --run-dir runs/sft_v3_large \
  --checkpoint best \
  --prompt "Hello, how are you?" \
  --max-tokens 100
```

---

## 2. Distillation v5 → v6 (Response-based avec Teacher)

### Étape 1: Installer mlx-lm
```bash
.venv/bin/pip install mlx-lm
```

### Étape 2: Tester avec un teacher rapide (Qwen2.5-1.5B)

Test avec 100 prompts pour vérifier la qualité :

```bash
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses_test.jsonl \
  --max-prompts 100 \
  --max-tokens 512 \
  --temperature 0.7
```

Vérifier la qualité des réponses :
```bash
head -n 3 data/sft/downloaded/raw/teacher_responses_test.jsonl
```

### Étape 3: Générer 10,000 réponses du teacher (5-10 heures)

Si le test est bon, lancer la génération complète :

```bash
nohup .venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7 \
  > logs/teacher_generation.log 2>&1 &
```

Surveiller la génération :
```bash
tail -f logs/teacher_generation.log
```

### Étape 4: Combiner avec les données existantes

Une fois la génération terminée :

```bash
# Créer le fichier combiné avec toutes les données
cat data/sft/downloaded/raw/combined_all.jsonl > data/sft/downloaded/raw/combined_with_teacher.jsonl
cat data/sft/downloaded/raw/teacher_responses.jsonl >> data/sft/downloaded/raw/combined_with_teacher.jsonl

# Vérifier le nombre total
wc -l data/sft/downloaded/raw/combined_with_teacher.jsonl
```

### Étape 5: Re-packer en v6

```bash
rm -rf data/sft/packed_v6

.venv/bin/python -m tinyllm.sft_data \
  --input data/sft/downloaded/raw/combined_with_teacher.jsonl \
  --tokenizer data/tokenizer.json \
  --output-dir data/sft/packed_v6 \
  --context-len 512
```

### Étape 6: Entraîner le student (TinyLLM) sur les données distillées

```bash
mkdir -p logs

nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v6/train.tokens.bin \
  --train-mask data/sft/packed_v6/train.mask.bin \
  --val-tokens data/sft/packed_v6/val.tokens.bin \
  --val-mask data/sft/packed_v6/val.mask.bin \
  --output-dir runs/sft_v6_distilled \
  --steps 3000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v6_distilled.log 2>&1 &
```

Surveiller l'entraînement :
```bash
tail -f logs/sft_v6_distilled.log
```

### Étape 7: Évaluer le modèle distillé

```bash
# Créer le config.json
cat > runs/sft_v6_distilled/config.json << 'EOF'
{
  "run_name": "sft_v6_distilled",
  "output_dir": "runs",
  "model": {
    "vocab_size": 16384,
    "d_model": 512,
    "n_layers": 12,
    "n_heads": 8,
    "n_kv_heads": 2,
    "ffn_hidden": 1408,
    "context_len": 512,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-05
  },
  "train": {
    "micro_batch": 4,
    "grad_accum": 8,
    "lr": 2e-05,
    "min_lr_ratio": 0.1,
    "warmup_ratio": 0.02,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "max_steps": 3000,
    "eval_interval": 100,
    "checkpoint_interval": 500,
    "log_interval": 10,
    "sample_interval": 1000,
    "max_hours": 48,
    "seed": 42,
    "dtype": "bfloat16"
  },
  "data": {
    "tokenizer_path": "data/tokenizer.json",
    "train_bin": "data/phase1/train.bin",
    "val_bin": "data/phase1/val.bin",
    "sources": {
      "tinystories": 1.0
    },
    "val_frac": 0.01,
    "test_frac": 0.01,
    "min_chars": 200,
    "max_chars": 100000,
    "test_bin": "data/phase1/test.bin"
  }
}
EOF

# Évaluer
.venv/bin/python -m tinyllm.sft_eval \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --sft-checkpoint runs/sft_v6_distilled/checkpoints/best \
  --test-tokens data/sft/packed_v6/test.tokens.bin \
  --test-mask data/sft/packed_v6/test.mask.bin

# Tester en chat interactif
.venv/bin/python -m tinyllm.chat \
  --run-dir runs/sft_v6_distilled \
  --checkpoint best
```

### Alternatives de Teacher Models

Si Qwen2.5-1.5B ne convient pas, essayer :

**Qwen2.5-0.5B-Instruct** (plus rapide, 2-4 heures) :
```bash
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model Qwen/Qwen2.5-0.5B-Instruct \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

**Gemma-2-2B-it** (plus lent, meilleure qualité, 10-20 heures) :
```bash
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model google/gemma-2-2b-it \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

---

## 3. Comparaison des modèles

Comparer les différentes versions :
- `runs/phase1_tinystories/checkpoints/best` - Modèle pré-entraîné (baseline)
- `runs/sft_v2_full/checkpoints/best` - SFT avec 35k conversations
- `runs/sft_v3_large/checkpoints/best` - SFT avec 164k conversations (en cours)
- `runs/sft_v4_teacher_responses/checkpoints/best` - SFT + distillation (futur)

```bash
# Comparaison rapide
for model in phase1_tinystories sft_v2_full sft_v3_large sft_v4_teacher_responses; do
  echo "=== $model ==="
  .venv/bin/python -m tinyllm.sft_eval \
    --base-config runs/phase1_tinystories/config.json \
    --base-checkpoint runs/phase1_tinystories/checkpoints/best \
    --sft-checkpoint runs/$model/checkpoints/best \
    --test-tokens data/sft/packed_v3/test.tokens.bin \
    --test-mask data/sft/packed_v3/test.mask.bin
done
```

---

## 4. Surveillance des entraînements

### Voir le progrès actuel
```bash
tail -f logs/sft_v3_large.log
```

### Voir les checkpoints sauvegardés
```bash
ls -lh runs/sft_v3_large/checkpoints/
```

### Voir les métriques récentes
```bash
tail -20 logs/sft_v3_large.log | jq '.'
```

---

## 5. Télécharger des données d'entraînement variées

### Télécharger des datasets depuis HuggingFace
```bash
# Script principal pour télécharger des datasets variés
.venv/bin/python download_sft_datasets.py \
  --max-general-fr 100000 \
  --max-coding-fr 50000 \
  --include-ultrachat \
  --ultrachat-limit 100000
```

### Télécharger OpenOrca séparément (si nécessaire)
```bash
.venv/bin/python download_openorca.py
```

### Combiner tous les datasets téléchargés
```bash
# Concaténer tous les fichiers JSONL dans le dossier raw
cat data/sft/downloaded/raw/*.jsonl > data/sft/downloaded/raw/combined_all.jsonl

# Vérifier le nombre total de conversations
wc -l data/sft/downloaded/raw/combined_all.jsonl
```

### Re-packer avec les nouvelles données variées
```bash
rm -rf data/sft/packed_v5

.venv/bin/python -m tinyllm.sft_data \
  --input data/sft/downloaded/raw/combined_all.jsonl \
  --tokenizer data/tokenizer.json \
  --output-dir data/sft/packed_v5 \
  --context-len 512
```

### Lancer un nouvel entraînement avec données variées (sans distillation)
```bash
mkdir -p logs

nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --val-tokens data/sft/packed_v5/val.tokens.bin \
  --val-mask data/sft/packed_v5/val.mask.bin \
  --output-dir runs/sft_v5_large \
  --steps 5000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v5_large.log 2>&1 &
```

### Surveiller l'entraînement
```bash
tail -f logs/sft_v5_large.log
```

---

## Notes

- **Modèle TinyLLM** : 42M paramètres, ~161 MB (model.safetensors)
- **v5 dataset** : 314,745 conversations, 193M tokens (3.3x plus que v3)
- **Temps d'entraînement** : ~2-3 heures pour 3000-5000 steps (indépendant de la taille des données)
- **Distillation** :
  - Qwen2.5-1.5B : 5-10 heures pour 10k prompts (recommandé)
  - Qwen2.5-0.5B : 2-4 heures pour 10k prompts (plus rapide)
  - Gemma-2-2B : 10-20 heures pour 10k prompts (meilleure qualité)
  - Gemma-3-4B : 25-50 heures pour 10k prompts (trop lent, déconseillé)
- **Workflow recommandé** :
  1. Tester Qwen2.5-1.5B avec 100 prompts
  2. Si qualité OK, générer 10,000 réponses
  3. Combiner avec v5, re-packer en v6
  4. Entraîner TinyLLM student sur v6
- **Alternative sans distillation** : Entraîner directement sur v5 (3.3x plus de données que v3)
- **Pour varier les sujets** : Utilisez `download_sft_datasets.py` pour télécharger des datasets HuggingFace (français, code, math, science, etc.)
