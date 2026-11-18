# Plan V9 — Modèle plus lourd, conversationnel FR/EN

> Statut : planifié, **à démarrer seulement après validation du SFT v8**.
> Objectif : un modèle ~86 M de paramètres, pré-entraîné sur ~2 milliards de tokens,
> puis fine-tuné pour converser naturellement en français (salutations, small talk,
> questions simples, refus honnêtes).

---

## 0. Pourquoi un V9 et pas un simple SFT v8

Un modèle plus lourd ne peut pas être obtenu en fine-tunant le checkpoint v8 :
la largeur et la profondeur sont fixées au pré-entraînement.

Le v8 actuel :

| Élément | Valeur |
|---|---|
| Paramètres | 46,4 M |
| Tokens de pré-entraînement | 616 M (~13 tok/param) |
| Tokens SFT supervisés | ~25 M (v8) / ~50 M (v8 conversationnel) |
| Temps de pré-entraînement | ~23 h |

Le V9 vise ~86 M de paramètres et ~2 milliards de tokens, soit ~23 tok/param,
ce qui est proche d'un budget raisonnable pour un petit modèle.

**Gain attendu** : meilleure modélisation de la langue, moins de répétitions,
meilleure tenue d'une conversation courte.
**Coût** : ~6,5 jours de pré-entraînement sur cette machine.

---

## 1. Architecture cible

| Paramètre | V8 | V9 |
|---|---:|---:|
| `vocab_size` | 24 576 | 24 576 |
| `d_model` | 512 | **640** |
| `n_layers` | 12 | **16** |
| `n_heads` | 8 | **10** |
| `n_kv_heads` | 2 | **2** |
| `ffn_hidden` | 1 408 | **1 792** |
| `context_len` | 512 | 512 |
| Paramètres estimés | 46,4 M | **~86 M** |

Le tokenizer reste **`data/v8/tokenizer.json`**. C'est volontaire :

- il est déjà optimisé pour le français (accents, ponctuation finale) ;
- il permet de **réutiliser le dataset SFT packé** sans re-tokenisation ;
- cela supprime un gros risque de régression.

---

## 2. Budget de données

Cible : **~2,0 milliards de tokens** après filtrage.

| Source | Tokens cibles | Part |
|---|---:|---:|
| `fr_wikipedia` | 450 M | 22,5 % |
| `fr_fineweb2` | 450 M | 22,5 % |
| `en_fineweb_edu` | 800 M | 40 % |
| `en_wikipedia` | 150 M | 7,5 % |
| `code_python` | 100 M | 5 % |
| `en_tinystories` | 10 M | 0,5 % |
| **Total** | **~1,96 Md** | 100 % |

Répartition : **~46 % FR**, **~49 % EN**, **~5 % code**.

Le corpus v8 faisait ~2,76 Go de texte brut pour 616 M de tokens.
Pour ~2 Md de tokens, compter **~9 Go de texte brut** et ~30 Go d'espace disque
avec les intermédiaires et les checkpoints.

---

## 3. Pipeline complet

> **Validé le 2026-11-12** : l'architecture V9 tourne sur la machine.
> Smoke test 20 steps sur les tokens v8 : **~4 000 tok/s**, pic mémoire **2,6 Go**,
> aucune divergence. À ce débit, 1,97 Md de tokens = **~5,7 jours**.
> Config prête dans `configs/pretrain_v9.json`.

### Étape 0 — Libérer de l'espace et figer le v8

1. Exécuter le SFT v8 conversationnel et noter les métriques de référence
   (`val_loss`, `loop_rate`, `distinct-1/2/3`, `OOV`, refus corrects).
2. Appliquer `archive_plan.md` pour libérer les checkpoints v1–v6.
3. Vérifier l'espace disque :

```bash
df -h .
du -sh runs/* data/v8 data/sft 2>/dev/null
```

### Étape 1 — Créer la config de pré-entraînement V9

```bash
cat > configs/pretrain_v9.json << 'EOF'
{
  "run_name": "pretrain_v9",
  "output_dir": "runs",
  "model": {
    "vocab_size": 24576,
    "d_model": 640,
    "n_layers": 16,
    "n_heads": 10,
    "n_kv_heads": 2,
    "ffn_hidden": 1792,
    "context_len": 512,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-05
  },
  "train": {
    "micro_batch": 2,
    "grad_accum": 64,
    "lr": 0.0005,
    "min_lr_ratio": 0.1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "max_steps": 30000,
    "eval_interval": 250,
    "checkpoint_interval": 500,
    "log_interval": 10,
    "sample_interval": 1000,
    "max_hours": 200.0,
    "seed": 42,
    "dtype": "bfloat16"
  },
  "data": {
    "tokenizer_path": "data/v8/tokenizer.json",
    "train_bin": "data/v9/pretrain/tokenized/train.bin",
    "val_bin": "data/v9/pretrain/tokenized/val.bin",
    "test_bin": "data/v9/pretrain/tokenized/test.bin",
    "sources": {
      "fr_wikipedia": 0.225,
      "fr_fineweb2": 0.225,
      "en_fineweb_edu": 0.40,
      "en_wikipedia": 0.075,
      "code_python": 0.05,
      "en_tinystories": 0.005
    },
    "val_frac": 0.015,
    "test_frac": 0.015
  }
}
EOF
```

Calcul du budget : `30 000 × 2 × 64 × 512 = 1,966 Md de tokens`.

### Étape 2 — Télécharger le corpus V9

`--char-scale 3.3` multiplie les budgets de `pretrain_v8.py` (2,76 Go)
pour viser ~9 Go de texte brut.

```bash
nohup .venv/bin/python scripts/download_pretrain_v8.py \
  --output-dir data/v9/pretrain/raw \
  --sources fr_wikipedia fr_fineweb2 en_fineweb_edu en_wikipedia en_tinystories code_python \
  --char-scale 3.3 \
  > logs/download_pretrain_v9.log 2>&1 &
```

Suivi :

```bash
tail -f logs/download_pretrain_v9.log
```

Durée estimée : plusieurs heures, selon le débit Hugging Face. Le script est
reprenable : relancer la même commande repart du dernier document complet.

### Étape 3 — Construire l'index anti-fuite

Empêche le corpus de pré-entraînement de contenir les prompts d'évaluation
ou les conversations SFT.

```bash
nohup .venv/bin/python scripts/build_leak_index_v8.py \
  --sft-file data/sft/downloaded/raw/combined_with_teacher.jsonl \
  --eval-dir data/eval \
  --output data/v9/pretrain/leak/leak_index.jsonl \
  > logs/build_leak_index_v9.log 2>&1 &
```

### Étape 4 — Filtrer le corpus

```bash
nohup .venv/bin/python scripts/filter_pretrain_v8.py \
  --input-dir data/v9/pretrain/raw \
  --output-dir data/v9/pretrain/filtered \
  --tokenizer data/v8/tokenizer.json \
  --leak-index data/v9/pretrain/leak/leak_index.jsonl \
  > logs/filter_pretrain_v9.log 2>&1 &
```

Le filtre applique : longueur, ratio alphabétique, détection de langue,
répétition, déduplication MinHash et anti-fuite.

### Étape 5 — Tokeniser

```bash
nohup .venv/bin/python scripts/tokenize_corpus.py \
  --input-dir data/v9/pretrain/filtered \
  --output-dir data/v9/pretrain/tokenized \
  --tokenizer data/v8/tokenizer.json \
  > logs/tokenize_pretrain_v9.log 2>&1 &
```

Vérifier ensuite le nombre réel de tokens dans
`data/v9/pretrain/tokenized/tokenize_stats.json`.

**Point de contrôle** : si le train fait moins de ~1,5 Md de tokens,
relancer le download avec un `--char-scale` plus élevé avant de continuer.

### Étape 6 — Pré-entraîner V9

```bash
mkdir -p logs
nohup .venv/bin/python -m tinyllm.train \
  --config configs/pretrain_v9.json \
  > logs/pretrain_v9.log 2>&1 &
```

Suivi :

```bash
tail -f logs/pretrain_v9.log
```

Durée estimée : **~6 à 7 jours** (débit attendu ~3 000–4 000 tok/s pour 86 M).
Sauvegardes : `runs/pretrain_v9/checkpoints/`.

### Étape 7 — SFT conversationnel V9

Le tokenizer est identique, donc `data/sft/packed_v8_conv` est réutilisable
tel quel. Créer la config du run :

```bash
mkdir -p runs/sft_v9
cp configs/sft_v8.json runs/sft_v9/config.json
sed -i '' 's/"run_name": "sft_v8"/"run_name": "sft_v9"/' runs/sft_v9/config.json
```

Puis :

```bash
nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config configs/pretrain_v9.json \
  --base-checkpoint runs/pretrain_v9/checkpoints/best \
  --train-tokens data/sft/packed_v8_conv/train.tokens.bin \
  --train-mask data/sft/packed_v8_conv/train.mask.bin \
  --val-tokens data/sft/packed_v8_conv/val.tokens.bin \
  --val-mask data/sft/packed_v8_conv/val.mask.bin \
  --replay-bin data/v9/pretrain/tokenized/train.bin \
  --replay-ratio 0.03 \
  --output-dir runs/sft_v9 \
  --steps 12000 \
  --micro-batch 4 \
  --grad-accum 8 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 500 \
  --log-interval 10 \
  --seed 42 \
  --dtype bfloat16 \
  > logs/sft_v9.log 2>&1 &
```

Durée estimée : ~10–12 h (modèle 2× plus lourd que V8).

### Étape 8 — Évaluation et comparaison

```bash
.venv/bin/python -m tinyllm.sft_eval \
  --base-config configs/pretrain_v9.json \
  --base-checkpoint runs/pretrain_v9/checkpoints/best \
  --sft-checkpoint runs/sft_v9/checkpoints/best \
  --test-tokens data/sft/packed_v8_conv/test.tokens.bin \
  --test-mask data/sft/packed_v8_conv/test.mask.bin
```

```bash
.venv/bin/python scripts/eval_quality.py \
  --run-dir runs/sft_v9 \
  --checkpoint best \
  --tokenizer data/v8/tokenizer.json \
  --prompts data/eval/quality_prompts.jsonl \
  --lexicon data/eval/fr_lexicon.txt \
  --output-json runs/sft_v9/eval_quality.json \
  --output-md runs/sft_v9/eval_quality.md
```

Chat :

```bash
.venv/bin/python -m tinyllm.chat \
  --run-dir runs/sft_v9 \
  --checkpoint best \
  --temperature 0.65 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

---

## 4. Critères go / no-go

### Après le pré-entraînement (étape 6)

| Critère | Seuil |
|---|---|
| Val loss V9 | ≤ 3,35 (V8 : 3,5967) |
| Échantillons à 25 %, 50 %, 75 % | français/anglais cohérent, pas de boucle |
| Tokens réellement vus | ≥ 1,9 Md |
| Aucun NaN / divergence | obligatoire |

Si la val loss stagne au-dessus de 3,5 après ~10 000 steps, arrêter et
revoir les données ou le taux d'apprentissage.

### Après le SFT (étape 8)

| Critère | Seuil |
|---|---|
| « Bonjour » → réponse de salutation | oui, sans boucle |
| « Comment ça va ? » | réponse cohérente |
| Refus hors périmètre | correct dans ≥ 80 % des cas |
| Fausses refus dans le périmètre | ≤ 10 % |
| `loop_rate` | ≤ 2 % |
| `OOV` FR | ≤ 3 % |

V9 n'est conservé que s'il bat V8 sur ces métriques. Sinon, garder V8.

---

## 5. Risques et parades

| Risque | Parade |
|---|---|
| Corpus insuffisant (< 1,5 Md tokens) | augmenter `--char-scale`, ajouter des sources FR |
| OOM pendant le pré-entraînement | `micro_batch: 2`, voire 1 ; `grad_accum` inchangé |
| Surapprentissage | 1 seul epoch sur ~2 Md tokens ; surveiller la val loss |
| Dérive anglaise | garder ≥ 45 % FR ; contrôler le mix dans `tokenize_stats.json` |
| Tokenizer inadapté | ne pas changer : réutiliser le tokenizer v8 |
| Checkpoints trop gros | `rotate_checkpoints` garde 3 + best (~3,5 Go) |
| Fuite eval → train | index anti-fuite obligatoire avant filtrage |
| Surchauffe / throttling Mac | surveiller `tokens_per_s`, éviter les autres tâches lourdes |

---

## 6. Checklist récapitulative

- [ ] V8 conversationnel entraîné et évalué (référence)
- [ ] `archive_plan.md` appliqué, ≥ 200 Go libres
- [ ] `configs/pretrain_v9.json` créé
- [ ] Corpus V9 téléchargé (~9 Go brut)
- [ ] Index anti-fuite V9 construit
- [ ] Corpus filtré
- [ ] Corpus tokenisé, ≥ 1,5 Md tokens vérifiés
- [ ] Pré-entraînement V9 lancé (~6,5 jours)
- [ ] Checkpoint intermédiaire à 25 % inspecté
- [ ] Val loss ≤ 3,35
- [ ] SFT V9 lancé (12 000 steps, données `packed_v8_conv`)
- [ ] Évaluation qualité V8 vs V9
- [ ] Décision : garder V9 ou rester sur V8
