# V8 — Phase d'annealing conversationnel

## Pourquoi

Le SFT principal (`runs/sft_v8_conv`, 12 000 steps) a appris le format
(`<|user|>…<|assistant|>`, réponses en français, arrêt sur EOS) mais pas les
réflexes courts : « bonjour » déclenche encore un paragraphe générique.

Cause : les salutations pèsent ~0,1 % de `packed_v8_conv`. Même sur 1,9 epoch,
chaque exemple est vu deux fois. C'est structurellement insuffisant.

L'annealing est un court passage à faible LR sur un dataset **entièrement court
et conversationnel**, qui réapprend ces réflexes sans casser le reste.

## Résultat du SFT principal

| Métrique | Valeur |
|---|---|
| Steps | 12 000 (7 h 48) |
| `val_loss` finale | 2,5599 |
| Meilleur checkpoint | `step_0011750` (`val_loss` 2,5593) |
| Courbe | 2,664 @ 3 000 → 2,592 @ 6 000 → 2,560 @ 12 000 |

Le SFT est convergé : le plateau commence vers 6 000.

## Contenu du dataset d'annealing

Builder : `scripts/build_annealing_v8.py`

| Bloc | Conversations |
|---|---:|
| Salutations (bonjour / salut / coucou / bonsoir) | 360 |
| « Comment ça va » | 180 |
| Identité (« qui es-tu ») | 144 |
| Capacités | 72 |
| Remerciements | 180 |
| Au revoir | 160 |
| Small talk (météo, humeur, blagues) | 98 |
| « Je ne sais pas » honnête | 53 |
| Salutations anglaises | 16 |
| Échanges multi-tours (2 et 3 tours) | 300 |
| Seed conversationnel réel | 153 |
| Replay FR réel (`general_fr`, réponses courtes, sans refus) | ~400 |
| **Total** | **~2 100** |

Soit ~2 550 séquences, ~49 000 tokens supervisés.

Toutes les réponses font 2 à 15 mots : le modèle désapprend le réflexe
« rédiger un article ».

Les réponses de type refus sont filtrées du replay : c'est ce qui produisait
l'attracteur « Je suis désolé, mais je ne peux pas… ».

## Commandes

### 1. Construire le dataset

```bash
.venv/bin/python scripts/build_annealing_v8.py \
  --tokenizer data/v8/tokenizer.json \
  --output-dir data/sft/packed_v8_anneal
```

Vérifier ensuite `data/sft/packed_v8_anneal/meta.json` : `totals.conversations`
doit être ~2 100 et `invariants` doit être à `true` partout.

### 2. Lancer l'annealing

```bash
.venv/bin/python -m tinyllm.sft_train \
  --base-config runs/pretrain_v8/config.json \
  --base-checkpoint runs/sft_v8_conv/checkpoints/best \
  --train-tokens data/sft/packed_v8_anneal/train.tokens.bin \
  --train-mask data/sft/packed_v8_anneal/train.mask.bin \
  --val-tokens data/sft/packed_v8_anneal/val.tokens.bin \
  --val-mask data/sft/packed_v8_anneal/val.mask.bin \
  --replay-bin data/v8/pretrain/tokenized/train.bin \
  --replay-ratio 0.05 \
  --output-dir runs/sft_v8_anneal \
  --steps 600 \
  --micro-batch 4 \
  --grad-accum 8 \
  --lr 8e-6 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --eval-interval 100 \
  --eval-batches 20 \
  --save-interval 100 \
  --log-interval 10 \
  --seed 42 \
  --dtype bfloat16
```

- 600 steps × 32 séquences = 19 200 séquences, soit **~7,6 epoch** du dataset.
- Durée : ~25 min.
- Départ depuis `runs/sft_v8_conv/checkpoints/best` (step 11 750), pas depuis
  `latest`.

### 3. Tester

#### Test reproductible (recommandé)

```bash
.venv/bin/python scripts/chat_regression.py \
  --run-dir runs/sft_v8_anneal \
  --checkpoint best \
  --json-out runs/sft_v8_anneal/regression.json
```

16 prompts fixes (salutations, identité, capacités, remerciements, au revoir,
heure, multi-tours), avec réponses attendues et réponses interdites. Le script
affiche un score et renvoie un code de sortie 1 sous `--min-score`.

Baselines mesurées :

| Run | Score |
|---|---:|
| `runs/sft_v8_conv` (SFT seul) | **0/16** |
| `runs/sft_v8_anneal` (passe 1) | **11/16** |

Échecs de la passe 1 : `capability`, `capability_casual`, `unknown_time`,
`multiturn_question_third`, `multiturn_identity_third`. Ce sont exactement les
défauts corrigés dans la passe 2.

Comparer plusieurs runs :

```bash
for run in runs/sft_v8_conv runs/sft_v8_anneal runs/sft_v8_anneal2; do
  echo "== $run"
  .venv/bin/python scripts/chat_regression.py --run-dir "$run" --json-out "$run/regression.json" | tail -3
done
```

#### Fuite de small talk (questions générales)

`chat_regression.py` mesure le comportement conversationnel. Il ne voit pas la
régression inverse : un modèle qui répond « Tout va bien, merci. » à une
question générale. `general_probe.py` comble ce trou.

```bash
.venv/bin/python scripts/general_probe.py \
  --run-dir runs/sft_v8_conv \
  --run-dir runs/sft_v8_anneal2
```

10 questions générales fixes (photosynthèse, repas, capitale, riz, poème,
arithmétique, résumé, vélo, langue, bibliothèque). Une réponse est marquée
`LEAK` si elle commence par une formule de small talk ou contient « merci de
demander », « et vous ? », « et toi ? ».

Baselines mesurées :

| Run | Fuites |
|---|---:|
| `runs/sft_v8_conv` | **0/10** |
| `runs/sft_v8_anneal2` | **3/10** |

Les deux métriques se lisent ensemble : un bon checkpoint maximise
`chat_regression` **et** minimise les fuites. Monter `--replay-ratio` réduit les
fuites mais peut faire perdre des points de conversation.

#### Test manuel

```bash
.venv/bin/python -m tinyllm.chat \
  --run-dir runs/sft_v8_anneal \
  --checkpoint best \
  --temperature 0.5 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.05
```

`/reset` entre chaque prompt :

```
bonjour
salut, comment ça va ?
qui es-tu ?
merci beaucoup
au revoir
que sais-tu faire ?
```

## Critères de réussite

| Test | Attendu |
|---|---|
| « bonjour » | « Bonjour ! » ou « Bonjour ! Comment puis-je vous aider ? » |
| « ça va ? » | « Je vais bien, merci ! Et vous ? » |
| « qui es-tu ? » | « Je suis TinyLLM, un petit modèle de langage. » |
| « merci » | « Avec plaisir ! » / « De rien ! » |
| « au revoir » | « À bientôt ! » / « Bonne journée ! » |
| réponse générique | plus de « Le code HTML5… » sur une salutation |

Si ces tests passent mais que les questions factuelles restent fausses, c'est
normal : l'annealing ne crée pas de connaissances. Voir `docs/V9_PLAN.md` ou la
prolongation du pré-entraînement.

## Si ça échoue

- **Salutations encore génériques à 600 steps** : vérifier que `best` est bien
  utilisé et que le prompt chat est `<|user|>bonjour<|assistant|>`. Puis
  descendre la température à 0,3.
- **Le modèle récite une seule réponse figée** : trop d'epochs. Reprendre à
  400 steps.
- **Régression générale** (le modèle ne sait plus répondre à autre chose) :
  monter `--replay-ratio` à 0,10 et relancer 400 steps.
- **Toujours mauvais après deux tentatives** : le plafond est le modèle de base
  (616 M tokens, `val_loss` 3,597). Passer à la prolongation du pré-entraînement
  ou à la V9.

## Limite connue

Ce modèle ne sait pas que Paris est la capitale de la France. L'annealing
corrige le comportement conversationnel, pas les connaissances. Cela se règle
au pré-entraînement.

## Résultats mesurés

| Run | `val_loss` | `chat_regression` | Fuites générales |
|---|---:|---:|---:|
| `sft_v8_conv` (SFT seul) | 2,5593 | **0/16** | 0/10 |
| `sft_v8_anneal` (passe 1, 600 steps) | 0,8691 | **11/16** | — |
| `sft_v8_anneal2` (passe 2, 500 steps) | 0,9887 | **15/16** → **17/21** | 3/10 |
| `sft_v8_anneal_v3` (profil v3, 400 steps) | 0,8385 | **18/21** | **1/10** |
| `sft_v8_anneal_v4` (profil v4, 400 steps) | 0,6461 | 18/21 | 2/10 |
| `sft_v8_anneal_v4b` (profil v4 corrigé, 400 steps) | 0,6916 | **20/21** | **1/10** |

### Le piège du gabarit

`sft_v8_anneal_v4` a donné exactement les mêmes échecs que `v3`, malgré un
dataset dont tous les pools `unknown_*` avaient été fusionnés. Cause : le
gabarit de dialogue long « calcul » contenait encore, en dur,
`"Je ne sais pas calculer de façon fiable, désolé."` — soit 61 conversations
qui réintroduisaient l'attracteur qu'on venait de supprimer des pools.

Leçon : auditer les **conversations générées**, pas seulement les définitions de
pools. Le profil `v4b` corrige le gabarit et ajoute un cas où « que sais-tu
faire ? » arrive en 3ᵉ position.

### Sensibilité à la graine

`v4b` passe 20/21 avec la graine par défaut, mais la gestion de l'inconnu reste
instable : sur 5 graines, « combien font 1+1 ? » donne 3 fois une réponse neutre
et 2 fois « Je ne connais pas la météo du jour, désolé. ». « calcule 5 + 3 »
part 4 fois sur 5 dans la blague aux poissons.

Autrement dit, le modèle choisit parmi un petit ensemble de phrases courtes
quand il est hors distribution. Supprimer une phrase déplace le problème vers
la suivante : c'est un plafond de capacité et de couverture, pas un bug de
dataset. `v4b` est le point d'arrêt raisonnable.


Le test est passé de 16 à 21 cas (ajout de dialogues libres, de `unknown_math`
et de deux cas de changement de sujet), donc les colonnes 16 et 21 ne se
comparent pas : c'est la paire `anneal2` / `anneal_v3` qu'il faut regarder.

### Le piège des scores trop beaux

`anneal_v3` a d'abord affiché 20/21. C'était faux : trois cas passaient par
accident, parce qu'une réponse « je ne sais pas » acceptée pour un thème
valide aussi pour un autre.

```
"quelle heure est-il ?"  -> "Je ne sais pas calculer de façon fiable, désolé."
"combien font 1+1 ?"     -> "Je ne connais pas la météo du jour, désolé."
```

Les listes d'attendus toléraient « désolé » sans vérifier le thème. Depuis, les
cas interdisent le mauvais thème (`calcul` pour l'heure, `météo` pour le
calcul). Score honnête : **18/21**.

### Migration de l'attracteur

Supprimer « je ne peux pas voir le ciel : je n'ai pas de capteurs » n'a pas
supprimé le problème : le modèle a promu « je ne sais pas calculer » au rang de
réponse universelle. Six pools `unknown_*` aux réponses quasi identiques
fusionnent forcément dans un modèle de 46 M.

D'où le profil `v4` : tous les `unknown_*` sont fusionnés en **un seul** pool
neutre, donc une réponse hors thème devient impossible par construction.


Le SFT seul n'a aucune fuite parce qu'il ne produit jamais de small talk : il
répond à côté. C'est donc `anneal2` qu'il faut comparer à `anneal3`, pas
`conv`.

### Conclusion de la passe 3

Doubler le replay (0,05 → 0,10) en baissant le LR ne change **rien** de
mesurable : mêmes 15/16, mêmes 3/10, mêmes réponses mot pour mot sur les
prompts qui fuient. Et les quatre checkpoints (`best`, 100, 200, 300) donnent
des scores identiques.

Deux conséquences :

1. La fuite de small talk n'est pas causée par un excès d'annealing. Elle est
   déjà dans le modèle de base : sur une question qu'il ne comprend pas, il
   complète avec ce qu'il a le plus vu dans le dataset — une formule de
   conversation.
2. Le levier n'est donc pas le replay, mais les données : supprimer les
   tournures qui servent d'attracteur et fournir une réponse neutre pour les
   questions hors compétence.

`anneal2` reste la référence : à score égal, elle est plus simple et n'a pas
coûté une passe supplémentaire.

Les `val_loss` ne sont pas comparables entre les passes : le split de
validation change avec le dataset, et la passe 3 repart d'un modèle déjà
ajusté. Seuls `chat_regression` et `general_probe` sont comparables.

Échec restant des passes 2 et 3 : `unknown_time` (« quelle heure est-il ? »
répond « Je ne peux pas voir le ciel : je n'ai pas de capteurs. »). La réponse
est honnête mais appartient au thème météo. Cause : la phrase « Je ne peux pas
voir le ciel : je n'ai pas de capteurs » est la seule réponse « honnête » du
pool `smalltalk_weather`, et elle devient la réponse par défaut à toute
question hors compétence (y compris « 1+1 »).

Note de mesure : le cas `multiturn_thanks_closes` acceptait « plaisir », « rien »,
« revoir », « bientôt », « journée » mais pas « je vous en prie », qui est une
réponse correcte. Corrigé — sans quoi la passe 3 était faussement notée 14/16.

