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

## 2. Distillation avec Gemma-3:4B (Response-based)

### Étape 1: Installer mlx-lm
```bash
.venv/bin/pip install mlx-lm
```

### Étape 2: Générer les réponses du teacher
```bash
# WARNING: Cette commande est lourde (~2-4 heures pour 10k prompts)
# Gemma-3:4B nécessite ~8-16 GB de RAM
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model google/gemma-3-4b-it \
  --train-tokens data/sft/packed_v3/train.tokens.bin \
  --train-mask data/sft/packed_v3/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

### Étape 3: Combiner avec les données existantes
```bash
# Ajouter les réponses du teacher aux données SFT existantes
cat data/sft/downloaded/raw/teacher_responses.jsonl >> data/sft/downloaded/raw/combined.jsonl
```

### Étape 4: Re-packer les données
```bash
# Supprimer l'ancien dossier packed s'il existe
rm -rf data/sft/packed_v4

# Re-packer avec les nouvelles données
.venv/bin/python -m tinyllm.sft_data \
  --input data/sft/downloaded/raw/combined.jsonl \
  --tokenizer data/tokenizer.json \
  --output-dir data/sft/packed_v4 \
  --context-len 512
```

### Étape 5: Entraîner avec les données augmentées
```bash
mkdir -p logs

nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v4/train.tokens.bin \
  --train-mask data/sft/packed_v4/train.mask.bin \
  --val-tokens data/sft/packed_v4/val.tokens.bin \
  --val-mask data/sft/packed_v4/val.mask.bin \
  --output-dir runs/sft_v4_teacher_responses \
  --steps 3000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v4_teacher_responses.log 2>&1 &
```

### Surveiller l'entraînement
```bash
tail -f logs/sft_v4_teacher_responses.log
```

### Étape 6: Évaluer le modèle distillé
```bash
# Créer le config.json
cat > runs/sft_v4_teacher_responses/config.json << 'EOF'
{
  "run_name": "sft_v4_teacher_responses",
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
  --sft-checkpoint runs/sft_v4_teacher_responses/checkpoints/best \
  --test-tokens data/sft/packed_v4/test.tokens.bin \
  --test-mask data/sft/packed_v4/test.mask.bin

# Tester en chat
.venv/bin/python -m tinyllm.chat \
  --run-dir runs/sft_v4_teacher_responses \
  --checkpoint best
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
# Dataset français général (conversations variées)
.venv/bin/python -m tinyllm.download_sft \
  --source huggingface \
  --dataset "HuggingFaceH4/no_robots" \
  --split train \
  --max-examples 100000 \
  --output data/sft/downloaded/raw/general_fr_v2.jsonl

# Dataset de code (Python, JavaScript, etc.)
.venv/bin/python -m tinyllm.download_sft \
  --source huggingface \
  --dataset "bigcode/self-oss-instruct-sc2-exec-filter-50k" \
  --split train \
  --max-examples 50000 \
  --output data/sft/downloaded/raw/code_v2.jsonl

# Dataset mathématiques et raisonnement
.venv/bin/python -m tinyllm.download_sft \
  --source huggingface \
  --dataset "meta-math/MetaMathQA" \
  --split train \
  --max-examples 50000 \
  --output data/sft/downloaded/raw/math.jsonl

# Dataset science (biologie, physique, chimie)
.venv/bin/python -m tinyllm.download_sft \
  --source huggingface \
  --dataset "openbookqa" \
  --split train \
  --max-examples 30000 \
  --output data/sft/downloaded/raw/science.jsonl

# Dataset questions-réponses variées
.venv/bin/python -m tinyllm.download_sft \
  --source huggingface \
  --dataset "Open-Orca/OpenOrca" \
  --split train \
  --max-examples 100000 \
  --output data/sft/downloaded/raw/qa_varied.jsonl
```

### Combiner tous les datasets téléchargés
```bash
# Concaténer tous les fichiers JSONL
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

### Lancer un nouvel entraînement avec données variées
```bash
mkdir -p logs

nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --val-tokens data/sft/packed_v5/val.tokens.bin \
  --val-mask data/sft/packed_v5/val.mask.bin \
  --output-dir runs/sft_v5_varied \
  --steps 5000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v5_varied.log 2>&1 &
```

### Surveiller l'entraînement
```bash
tail -f logs/sft_v5_varied.log
```

---

## Notes

- Le modèle pèse ~161 MB (model.safetensors)
- L'entraînement v3 devrait finir dans ~1.7 heures
- La distillation avec Gemma-3:4B nécessite ~8-16 GB de RAM
- Les commandes de distillation sont facultatives - tester d'abord le modèle v3
- Pour varier les sujets, téléchargez différents datasets HuggingFace (code, math, science, etc.)
