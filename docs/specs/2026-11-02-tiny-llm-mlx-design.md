# Tiny LLM from scratch en MLX — Spécification de conception

- **Date** : 2026-11-02
- **Statut** : approuvé pour planification
- **Auteur** : conception collaborative (utilisateur + agent)

## 1. Résumé

Construire de zéro, en MLX, un petit modèle de langage decoder-only destiné à
l'apprentissage : tokenizer, pipeline de données, architecture, boucle
d'entraînement, évaluation, quantification et inférence. Le projet se déroule
en deux étapes distinctes :

- **Phase 1 — validation** : un petit modèle (~42 M) entraîné sur TinyStories,
  dont le seul rôle est de valider le pipeline et de produire rapidement des
  échantillons lisibles.
- **Phase 2 — run final** : un modèle plus grand (~80 M, extensible à ~110 M)
  entraîné **from scratch** sur un corpus mixte anglais / français / code.

La phase 2 ne réutilise pas les poids de la phase 1 : les architectures diffèrent
et un transfert de poids n'apporterait rien à cette échelle. La phase 1 est un
investissement de débogage, pas une initialisation.

Le livrable final est un modèle d'environ **80 M paramètres** (extensible à
~110 M), quantifié en 4-bit, dont les **poids pèsent moins de 100 Mo**, exécutable
localement sur un Mac Apple Silicon.

## 2. Objectifs et non-objectifs

### Objectifs

1. Écrire l'intégralité du modèle et de l'entraînement à la main en MLX pur
   (pas de `mlx-lm` pour le modèle ni pour la boucle d'entraînement).
2. Produire un modèle qui génère de l'anglais grammatical et cohérent, avec des
   bases en français et en code Python.
3. Rendre chaque étape mesurable : bits-per-byte, perplexité par domaine,
   échantillons qualitatifs reproductibles.
4. Livrer un pipeline reproductible : une commande prépare les données, une
   commande entraîne, une commande quantifie, une commande génère.
5. Documenter honnêtement les limites du modèle.

### Non-objectifs (assumés)

- Pas d'assistant conversationnel généraliste, pas de suivi d'instructions
  fiable, pas de raisonnement multi-étapes.
- Pas de RLHF, pas de DPO, pas de tool use.
- Pas de serving, pas d'API HTTP, pas d'interface graphique.
- Pas de support multi-GPU / multi-machine.
- Pas de compatibilité avec les formats `mlx-lm` en entrée ; l'export vers un
  format standard est un stretch goal.

## 3. Contrainte mémoire : deux cibles distinctes

La demande initiale « ~100 Mo en RAM » recouvre deux mesures différentes. La
spec retient la première et documente la seconde.

| Mesure | Cible | Réalité attendue |
|---|---|---|
| Poids du modèle quantifiés 4-bit | **≤ 100 Mo** | 40-55 Mo pour ~80-110 M params |
| Poids en fp16 (avant quantification) | — | 160-230 Mo |
| RAM totale du process (RSS) à l'inférence | non contrainte | 300-600 Mo (Python + MLX + Metal) |
| RAM d'entraînement | ≤ 16 Go | ~6-10 Go sur Mac Mini M4 |

**Décision** : « 100 Mo » désigne les poids du modèle. Atteindre 100 Mo de RSS
total exigerait un runtime Swift/C++ et un modèle d'environ 15 M paramètres,
ce qui constitue un projet différent.

Le rapport final mesurera et publiera les trois chiffres (poids, pic mémoire
Metal, RSS) sans les confondre.

## 4. Matériel et environnement

- **Entraînement** : Mac Mini M4, 16 Go de mémoire unifiée.
- **Inférence / développement** : MacBook M3, 8 Go.
- **OS** : macOS récent, Apple Silicon uniquement.
- **Python** : 3.11+, géré par `uv`.
- **Dépendances principales** : `mlx`, `numpy`, `tokenizers`, `datasets`,
  `huggingface-hub`, `pytest`. Versions épinglées dans `pyproject.toml`.
- **Espace disque** : le sous-ensemble de corpus visé doit tenir dans ~30 Go
  (86 Go libres sur la machine de développement). Les datasets sont streamés et
  filtrés, pas téléchargés intégralement quand leur taille le permet.

## 5. Architecture du modèle

Transformer decoder-only, pre-norm, style Llama, implémenté en MLX.

### Composants

- **Embedding** de tokens, sans biais, lié à la projection de sortie
  (*weight tying*).
- **RMSNorm** pre-norm avant attention et avant MLP.
- **Attention causale** avec **RoPE** (theta = 10 000) et **GQA**
  (grouped-query attention).
- **MLP SwiGLU** : `down(silu(gate(x)) * up(x))`.
- **Pas de biais** dans les couches linéaires.
- **Cache KV** pour la génération incrémentale.
- Dropout à 0,0 (corpus volumineux, modèle sous-paramétré).

### Deux configurations

| Paramètre | Phase 1 | Phase 2 (défaut) | Phase 2 (étendu) |
|---|---|---|---|
| `d_model` | 512 | 640 | 768 |
| Couches | 12 | 16 | 16 |
| Têtes Q | 8 | 10 | 12 |
| Têtes KV | 2 | 2 | 4 |
| `head_dim` | 64 | 64 | 64 |
| FFN hidden | 1408 | 1792 | 2048 |
| Vocab | 16 384 | 16 384 | 16 384 |
| Contexte | 512 | 512 | 512 |
| Params approx. | ~42 M | ~80 M | ~110 M |
| Poids 4-bit | ~21 Mo | ~40 Mo | ~55 Mo |

Le contexte de 512 tokens est un choix assumé : à cette échelle, allonger le
contexte augmente le coût quadratique sans bénéfice mesurable sur les objectifs.

### Interface

```python
class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def __call__(self, tokens: mx.array, cache: list[KVCache] | None = None) -> mx.array: ...
```

- `tokens` : `int32`, shape `(batch, seq_len)`.
- Sortie : logits `float32` de shape `(batch, seq_len, vocab)`.
- `cache` : liste par couche, `None` en entraînement.

## 6. Tokenizer

- **BPE byte-level**, vocabulaire de **16 384** tokens.
- Entraîné avec la bibliothèque `tokenizers` sur un échantillon de 2-4 Go du
  corpus final (mélange des sources), pas sur TinyStories seul.
- Tokens spéciaux :
  - `<|endoftext|>` (id 0) : fin de document, utilisé aussi comme séparateur de
    packing.
  - `<|pad|>` réservé, non utilisé en entraînement.
  - `<|user|>` et `<|assistant|>` réservés pour un éventuel SFT ultérieur,
    jamais vus en pré-entraînement.
- Justification du vocabulaire réduit : un vocab de 50 480 (GPT-2) coûterait
  ~32 M paramètres en embeddings à `d_model=640`, soit environ 30 % du budget
  total du modèle.

## 7. Données

### Sources

| Phase | Source | Part | Usage |
|---|---|---|---|
| 1 | `roneneldan/TinyStories` | 100 % | anglais simple, cohérence |
| 2 | `HuggingFaceFW/fineweb-edu` (échantillon filtré) | 65 % | connaissances générales EN |
| 2 | `wikimedia/wikipedia` 20231101.en (sous-ensemble) | 10 % | faits EN |
| 2 | `wikimedia/wikipedia` 20231101.fr (sous-ensemble) | 10 % | français |
| 2 | `bigcode/the-stack-smol` (Python) | 10 % | code |
| 2 | `roneneldan/TinyStories` | 5 % | préserve la cohérence de l'anglais simple |

Les sources sont publiques et largement permissives ; les licences sont
documentées dans `docs/data-sources.md`. Aucun corpus propriétaire.

### Nettoyage

1. Filtre de longueur : documents entre 200 et 100 000 caractères.
2. Filtre de langue (détection légère par stopwords ou `fasttext` si disponible).
3. Suppression HTML/Markdown résiduel, normalisation des espaces.
4. Déduplication exacte par hash ; déduplication floue (MinHash) en option si
   le volume le justifie.
5. Filtre qualité simple pour le code : pas de fichier généré, pas de minified.

### Packing et format

- Documents concaténés avec `<|endoftext|>` entre eux.
- Découpage en séquences de 512 tokens ; le dernier chunk incomplet est jeté.
- Stockage : un fichier binaire `uint16` par split (`train.bin`, `val.bin`,
  `test.bin`) + `meta.json` (nombre de tokens, hash du tokenizer, config).
- Chargement par `numpy.memmap` puis conversion en `mx.array` par batch.
- **Split par document**, jamais par chunk : un document ne peut pas apparaître
  dans train et dans val.
- Volumes cibles : Phase 1 ~500 M tokens ; Phase 2 ~2 B tokens vus, corpus
  préparé de 4-6 B tokens pour permettre plusieurs epochs sans répétition
  immédiate.

## 8. Boucle d'entraînement

### Hyperparamètres par défaut

| Paramètre | Phase 1 | Phase 2 |
|---|---|---|
| Optimiseur | AdamW | AdamW |
| β | (0,9 ; 0,95) | (0,9 ; 0,95) |
| Weight decay | 0,1 | 0,1 |
| Learning rate max | 6e-4 | 3e-4 |
| Warmup | 2 % des steps | 2 % des steps |
| Décroissance | cosinus → 10 % du max | cosinus → 10 % du max |
| Gradient clipping | 1,0 | 1,0 |
| Micro-batch | 16 × 512 | 16 × 512 |
| Accumulation | ×8 | ×8 |
| Tokens / step | 65 536 | 65 536 |
| Steps | ~7 600 (500 M tokens) | ~30 500 (2 B tokens) |
| Précision calcul | bf16 | bf16 |
| États optimiseur | fp32 | fp32 |

### Détails d'implémentation

- Loss : entropie croisée sur les tokens décalés, `ignore_index` non utilisé
  (pas de padding grâce au packing).
- `mx.value_and_grad` sur une fonction de loss pure ; accumulation manuelle des
  gradients avant `optimizer.update`.
- Paramètres en bf16 pour le forward, copie fp32 pour l'optimiseur.
- Seed fixée et propagée (init, shuffling, sampling).
- Checkpoint toutes les 500 steps dans `runs/<run_id>/checkpoints/` :
  poids (`safetensors`), état optimiseur, step, meilleure val loss, config
  complète, état RNG. Reprise exacte via `--resume`.
- Rotation des checkpoints : garder les 3 derniers + le meilleur.
- Logs JSONL : `step`, `tokens_seen`, `loss`, `lr`, `grad_norm`, `tokens/s`,
  `peak_memory_mb` toutes les 10 steps ; val loss toutes les 500.
- Échantillons générés toutes les 1 000 steps sur 3 prompts fixes, sauvegardés
  dans `runs/<run_id>/samples/step_XXXXX.txt`.
- Arrêt propre sur SIGINT : checkpoint puis sortie.
- **Garde-fou mémoire** : si le pic dépasse 13 Go, réduire le micro-batch et
  augmenter l'accumulation pour conserver 65 536 tokens/step.

### Contrôle du budget

La vitesse réelle est mesurée sur les 100 premiers steps. Le script refuse de
démarrer un run dont l'estimation dépasse la durée maximale déclarée dans la
config (`max_hours`), sauf flag `--force`.

## 9. Évaluation

### Métriques quantitatives

- **Bits-per-byte (BPB)** sur `val.bin`, calculé comme
  `loss_nats / (ln(2) × octets)`. Métrique principale, comparable entre
  tokenizers et entre phases.
- **Perplexité par domaine** sur des splits de validation dédiés : EN général,
  FR, code Python.
- **Accuracy next-token** sur un échantillon, comme sanity check secondaire.

### Métriques qualitatives

- Suite de 8 prompts fixes (débuts de phrases EN, 2 prompts FR, 2 prompts code),
  générés à température 0,7 et 1,0, top-p 0,9, 200 tokens.
- Grille de comparaison phase 1 / phase 2 / checkpoint intermédiaire, archivée
  dans le rapport final.
- Grille de lecture humaine : grammaticalité, cohérence locale, cohérence
  globale, répétitions. Échelle 1-5, remplie par l'utilisateur, sans prétention
  de benchmark.

### Tests de sanité obligatoires

1. **Overfit d'un batch** : 300 steps sur un batch unique, avec la config
   Phase 1, doivent amener la loss sous 0,1.
2. **Masque causal** : modifier les tokens futurs ne doit pas changer les
   logits présents (à 1e-5 près).
3. **Équivalence cache KV** : prefill + décodage incrémental doit produire les
   mêmes logits qu'un forward complet (atol 1e-4).
4. **Roundtrip tokenizer** : `decode(encode(x)) == x` sur un échantillon de
   texte, hors normalisation d'espaces documentée.
5. **Intégrité du packing** : aucun chunk ne mélange deux documents sans
   `<|endoftext|>` ; les splits ne partagent aucun document.

## 10. Inférence et quantification

- Génération avec cache KV, température, top-k, top-p et pénalité de répétition.
- `sample.py` : complétion de texte à partir d'un prompt (pas de chat).
- `quantize.py` : quantification 4-bit, group size 64, via `mx.quantize`,
  sauvegarde en `safetensors` + `config.json`. Vérifie que la perplexité
  dégradée reste sous un seuil déclaré (par défaut +15 % de BPB).
- Mesures publiées : taille des poids sur disque, pic mémoire Metal
  (`mx.metal.get_peak_memory()`), RSS (`/usr/bin/time -l`).
- Stretch : export vers un format lisible par `mlx-lm` pour comparaison.

## 11. Structure du dépôt

```
LLM/
├── pyproject.toml
├── README.md
├── configs/
│   ├── phase1_tinystories.json
│   ├── phase2_mix.json
│   └── phase2_mix_extended.json
├── src/tinyllm/
│   ├── __init__.py
│   ├── config.py        # dataclasses ModelConfig / TrainConfig / DataConfig
│   ├── tokenizer.py     # entraînement + encode/decode
│   ├── data.py          # téléchargement, nettoyage, packing, dataset
│   ├── model.py         # TinyLM, blocs, RoPE, GQA, cache KV
│   ├── train.py         # boucle, optimiseur, checkpoints, logs
│   ├── sample.py        # génération CLI
│   ├── quantize.py      # quantification + vérification
│   └── eval.py          # BPB, perplexité par domaine, échantillons
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   ├── run_phase1.sh
│   └── run_phase2.sh
├── tests/
│   ├── test_model_shapes.py
│   ├── test_causal_mask.py
│   ├── test_kv_cache.py
│   ├── test_tokenizer.py
│   ├── test_packing.py
│   └── test_overfit_batch.py   # marqué slow
├── docs/
│   ├── specs/2026-11-02-tiny-llm-mlx-design.md
│   └── data-sources.md
└── runs/                # gitignoré
```

## 12. Interface en ligne de commande

```bash
# 1. Tokenizer
uv run python scripts/train_tokenizer.py --config configs/phase1_tinystories.json

# 2. Données
uv run python scripts/prepare_data.py --config configs/phase1_tinystories.json

# 3. Entraînement
uv run python -m tinyllm.train --config configs/phase1_tinystories.json

# 4. Reprise
uv run python -m tinyllm.train --config configs/phase1_tinystories.json \
  --resume runs/phase1/checkpoints/latest

# 5. Évaluation
uv run python -m tinyllm.eval --checkpoint runs/phase1/checkpoints/best \
  --split val

# 6. Quantification
uv run python -m tinyllm.quantize --checkpoint runs/phase2/checkpoints/best \
  --bits 4 --group-size 64 --out models/tinyllm-80m-4bit

# 7. Génération
uv run python -m tinyllm.sample --model models/tinyllm-80m-4bit \
  --prompt "Once upon a time" --max-tokens 200 --temperature 0.7
```

Chaque commande échoue avec un message clair si une étape précédente manque
(tokenizer absent, données absentes, checkpoint incompatible).

## 13. Tests et qualité

- `pytest` pour tous les tests unitaires ; `test_overfit_batch.py` marqué
  `slow`, exécuté manuellement.
- Formatage : `ruff format` + `ruff check`.
- Typage léger : annotations partout, `mypy` en option non bloquante.
- Aucun test ne dépend du réseau (fixtures locales miniatures).
- Chaque script est idempotent : relancer `prepare_data.py` ne retélécharge pas
  ce qui existe déjà et vérifie les hashes.

## 14. Jalons et critères d'acceptation

| Jalon | Contenu | Critère d'acceptation | Durée |
|---|---|---|---|
| **M0** | Repo, config, tokenizer, data, modèle, tests | `pytest` vert, overfit-batch passe, `prepare_data` produit `train.bin`, `docs/data-sources.md` rédigé | 1-2 j |
| **M1** | Run Phase 1 + évaluation | BPB val < 1,2 sur TinyStories (indicatif), échantillons anglais lisibles | 1-2 j calcul |
| **M2** | Corpus Phase 2 + run from scratch | BPB val < 1,1 sur le mix (indicatif), texte EN cohérent, FR et code non triviaux | 4-7 j calcul |
| **M3** | Quantification + CLI | Poids ≤ 100 Mo, dégradation BPB < 15 %, génération fonctionnelle | 1 j |
| **M4** | Documentation + rapport | README reproductible, rapport avec courbes, échantillons, mesures mémoire | 1 j |

Les seuils BPB de M1 et M2 sont indicatifs : ils seront recalibrés après la
première mesure réelle, mais toute révision sera consignée dans la spec.

## 15. Risques et mitigations

| Risque | Impact | Mitigation |
|---|---|---|
| OOM sur 16 Go | run interrompu | micro-batch 8, accumulation ×16 ; gradient checkpointing en secours |
| Pipeline buggé découvert tard | jours perdus | phase 1 de validation + tests de sanité avant tout run long |
| Données indisponibles / licences | blocage | sources de repli documentées, cache local, sous-ensembles |
| Disque insuffisant | blocage | streaming HF, filtrage à la volée, corpus plafonné à 30 Go |
| Tokenizer de mauvaise qualité | texte dégradé | entraîner sur un échantillon du mix final, vérifier le roundtrip et le taux de caractères par token |
| Catastrophic forgetting en phase 2 | perte de l'anglais simple | mélanger TinyStories à 5 % dans le corpus phase 2 |
| Débit réel inférieur à l'estimation | run trop long | mesure sur 100 steps, garde-fou `max_hours`, arrêt et reprise |
| Confusion mémoire 100 Mo | attentes déçues | trois mesures publiées séparément (poids, Metal, RSS) |

## 16. Reproductibilité

- Versions épinglées, seed documentée, hash du tokenizer et hash des données
  stockés dans chaque run.
- Un run est décrit par un seul `config.json` copié dans son dossier.
- Commandes exactes consignées dans `runs/<run_id>/command.txt`.
- Aucun résultat rapporté sans le checkpoint et la config associés.

## 17. Définition de « terminé »

Le projet est terminé quand :

1. `pytest` passe sur une machine propre après `uv sync`.
2. Le pipeline complet s'exécute de bout en bout avec les commandes du §12.
3. Un modèle quantifié 4-bit de poids ≤ 100 Mo génère du texte à partir d'un
   prompt, avec un BPB documenté.
4. Le rapport final contient les courbes de loss, les échantillons, les mesures
   mémoire et une section explicite sur les limites.
