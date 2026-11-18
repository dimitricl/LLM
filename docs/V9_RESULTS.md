# V9 — Résultats

> Statut : **terminé** (18 novembre 2026). Pré-entraînement, SFT conversationnel (arrêté
> sur plateau), annealing, version Jarvis avec outils (v1 et v2) : voir la section 8.
> Le modèle maison a atteint la limite de cette configuration (86 M de paramètres, 2 Md de tokens).

## 1. Configuration

| Élément | Valeur |
|---|---|
| Paramètres | 86 528 640 |
| Couches / d_model / têtes | 16 / 640 / 10 (GQA 10:2) |
| FFN | 1 792 |
| Contexte | 512 |
| Tokenizer | identique à V8 (`data/v8/tokenizer.json`) |
| Tokens de pré-entraînement | 1 966 080 000 |
| Ratio tokens / paramètres | 23 |

Config de référence : `configs/pretrain_v9.json`.

## 2. Pré-entraînement

`runs/pretrain_v9` — 30 000 steps, 5 j 15 h 53, sans incident.

```
débit moyen        4 019 tok/s  (3 938 – 4 086, aucune dérive)
grad_norm          0,45 stable de bout en bout
mémoire de pic     2 571 Mo, constante
loss finale        3,124
```

Courbe de validation (corpus V9) :

| step | val_loss |
|---|---|
| 4 000 | 3,790 |
| 6 500 | 3,611 |
| 16 000 | 3,330 |
| 25 000 | 3,195 |
| 27 000 | 3,1787 |
| 30 000 | **3,1587** |

Les `val_loss` V8 et V9 ne sont pas comparables : les corpus de validation
diffèrent. Seuls le banc conversationnel et le probe de fuites le sont.

## 3. SFT d'annealing

`runs/sft_v9_anneal_a` — 800 steps depuis `runs/pretrain_v9/checkpoints/best`,
dataset `data/sft/packed_v8_anneal_v4b` (5 324 séquences, 4,5 époques),
replay 5 % du corpus de pré-entraînement, lr 1e-5 cosine.

```
durée      53 min 42 s
débit      4 072 tok/s
val_loss   1,2539 (step 100) -> 0,6538 (step 800), décroissance monotone
```

### Banc 21 cas (`scripts/chat_regression.py`)

| graine | score | échec |
|---|---|---|
| 42 | 21/21 | — |
| 1 | 20/21 | mot-clé « fiable » absent du barème |
| 2 | 21/21 | — |
| 3 | 21/21 | — |
| 4 | 20/21 | « Y a pas de quoi ! » absent du barème |

**103/105 brut, 105/105 en sémantique.** Les deux échecs sont des trous du
barème, pas des fautes du modèle.

### Probe de fuites (`scripts/general_probe.py`)

**0/10** réponses small talk sur des questions générales (V8 : 1/10).

### Comparaison V8 / V9

| | V8 (`sft_v8_anneal_v4b`, 46 M) | V9 (`sft_v9_anneal_a`, 86 M) |
|---|---|---|
| Banc 21 cas | 18–20/21 selon la graine | 21/21 |
| Fuites small talk | 1/10 | 0/10 |
| « combien font 1+1 ? » | « Je ne connais pas la météo du jour » | « Je ne sais pas » |
| « calcule 5 + 3 » | blague aux poissons | « Je ne sais pas » |
| Passes nécessaires | 5 passes d'annealing | 1 passe |

Les deux attracteurs structurels du V8 — le gabarit de calcul qui réinjectait
la météo, et le glissement vers la blague — ont disparu.

## 4. Limites connues

1. **Sur-refus sur les accusés de réception.** `d'accord.` déclenche
   « Je ne sais pas » au lieu d'une relance. Cause : `d'accord` n'existe dans
   le corpus qu'en contexte de clôture (`build_annealing_v8.py:789`), et le
   pool `NEUTRAL_UNKNOWN_REPLIES` couvre trop large.
2. **Pas de cas « informations sur l'utilisateur ».** « comment je m'appelle ? »
   retombe sur le pool identité et dérive. Aucune entrée pour nom, âge, ville.
3. **Identité instable.** Trois descriptions différentes en quatre tours
   (`IDENTITY_REPLIES` a trop de variantes, `temp=0.65` les fait alterner).
   Certaines citent « 46 millions de paramètres », chiffre hérité du V8.
4. **Connaissances factuelles faibles.** « capitale de la France » → Kinshasa,
   photosynthèse → charabia. C'est de la capacité : 86 M de paramètres ne
   peuvent pas stocker un corpus encyclopédique. Aucun SFT ne corrige cela.
5. **Surapprentissage du dataset d'annealing.** `val_loss` 0,65 sur 4,5 époques
   de 5 324 séquences = réponses en formules. D'où le SFT conversationnel
   complet sur `packed_v8_conv` (201 000 séquences), en cours.

## 5. Reproductibilité

```bash
# pré-entraînement
python -m tinyllm.train --config configs/pretrain_v9.json

# SFT d'annealing
python -m tinyllm.sft_train \
  --base-config runs/pretrain_v9/config.json \
  --base-checkpoint runs/pretrain_v9/checkpoints/best \
  --train-tokens data/sft/packed_v8_anneal_v4b/train.tokens.bin \
  --train-mask   data/sft/packed_v8_anneal_v4b/train.mask.bin \
  --val-tokens   data/sft/packed_v8_anneal_v4b/val.tokens.bin \
  --val-mask     data/sft/packed_v8_anneal_v4b/val.mask.bin \
  --replay-bin data/v9/pretrain/tokenized/train.bin --replay-ratio 0.05 \
  --output-dir runs/sft_v9_anneal_a \
  --steps 800 --micro-batch 4 --grad-accum 8 --lr 1e-5 \
  --seed 42 --dtype bfloat16

# évaluation
python scripts/chat_regression.py --run-dir runs/sft_v9_anneal_a --seed 42
python scripts/general_probe.py --run-dir runs/sft_v9_anneal_a
```

## 6. Fichiers

| Chemin | Contenu |
|---|---|
| `runs/pretrain_v9/checkpoints/best` | base V9 (86,5 M, 346 Mo fp32) |
| `runs/sft_v9_anneal_a/checkpoints/best` | meilleur modèle conversationnel actuel |
| `runs/sft_v9_anneal_a/regression_s*.json` | résultats des 5 graines |
| `runs/pretrain_v9/log.jsonl` | courbe de pré-entraînement |
| `data/sft/packed_v8_anneal_v4b` | dataset d'annealing (8,3 Mo) |
| `data/sft/packed_v8_conv` | corpus conversationnel complet (305 Mo, 201 k séq.) |
| `data/v9/pretrain/tokenized/train.bin` | corpus de pré-entraînement (replay) |

## 7. Suite (historique)

Le plan initial prévoyait un SFT conversationnel complet puis un annealing « v5 ». Réalisé
autrement : le SFT conv a été arrêté au step 3200 (plateau, 11/45 au banc neutre, perte des
réflexes courts) ; l'annealing final est `sft_v9_jarvis2` (section 8).

## 8. Bilan final (18 novembre 2026)

### Chaîne réalisée
Pré-entraînement V9 (30 000 steps, val_loss 3,1587) -> `sft_v9_anneal_a` -> `sft_v9_jarvis`
(persona Jarvis + outils) -> **`sft_v9_jarvis2`** (outils corrigés + `<think>` programmatique,
600 steps, val_loss 0,457 -> 0,293). Détails : `docs/JARVIS_V9_V2.md`.

### Banc neutre (45 cas, `scripts/bench_neutral.py`, un tirage)
| Modèle | Score |
|---|---|
| V9 `anneal_a` | 12 |
| V9 conv (step 3000) | 11 |
| V9 lite / tiny (exports) | 12 / 9 |
| V9 Jarvis v1, avec outils | 21 |
| **V9 Jarvis v2, avec outils** | **20** |
| LFM2.5-230M 8 bits / 350M 4 bits / 350M 8 bits | 35 / 33 / 39 |
| LFM3-1.7B 4 bits (base / + LoRA Jarvis) | 42 / 41 |

### Autres mesures V9 Jarvis v2
| Mesure | Résultat |
|---|---|
| Banc de régression 21 cas | 18/21 (v1 : 16/21 ; `anneal_a` : 21/21, ancien jeu de mots-clés) |
| Recopie d'arithmétique, 40 questions nouvelles | 28/40 (v1 : 14/40) |
| Date (texte cohérent avec l'outil) | correcte (v1 : fausse) |
| Raisonnement, formulation d'entraînement | 41/63 |
| Raisonnement, formulations reformulées | 6 à 10/63 |
| Vitesse | ~220 tok/s (MacBook Air, mesure utilisateur) ; 260-430 tok/s (Mac mini) |
| RAM du processus | 159 Mo mesurés sur l'Air (bundle conv, même format) ; ~190 Mo (Mac mini, `lite`) |

### Conclusion
- Fort : conversation courte, persona Jarvis stable, refus corrects (4/4 en ignorance),
  calcul par outil, vitesse. Projet complet de bout en bout (tokenizer, pré-entraînement,
  SFT, quantification, serveur avec outils, bancs d'évaluation).
- Limites mesurées : aucune connaissance factuelle fiable (faits 0 à 1/9), le `<think>`
  ne généralise pas (templates), recopie de nombres imparfaite, RAM au-dessus de 100 Mo.
- Aller plus loin demanderait plus de tokens et de paramètres (semaines de calcul) ou un
  modèle parent ; la voie capable est Jarvis sur LFM3 (branche `feat/jarvis-lfm3`).
