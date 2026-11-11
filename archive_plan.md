# Plan d'archivage disque

## Résumé
- **Espace total actuel**: ~60 Go
- **Espace gardé**: ~20 Go
- **Espace libéré**: ~40 Go

## Tableau des éléments à archiver

| Chemin | Taille | Raison | Action |
|--------|--------|--------|--------|
| runs/pretrain_v8/checkpoints/step_0004500 | 531 Mo | Checkpoint intermédiaire | Archiver |
| runs/pretrain_v8/checkpoints/step_0004600 | 531 Mo | Checkpoint intermédiaire | Archiver |
| runs/pretrain_v8/checkpoints/step_0004700 | 531 Mo | Checkpoint final (latest en pointe vers lui) | Garder (ou supprimer si latest pointe vers best) |
| runs/phase1_tinystories/checkpoints/* sauf best | ~2.4 Go | Ancien run, seul best utile | Archiver les checkpoints intermédiaires |
| runs/sft_v1/checkpoints/* sauf best | ~2.4 Go | Version v1 obsolète | Archiver tout (y compris best ou archiver best séparément) |
| runs/sft_v2_full/checkpoints/* sauf best | ~2.4 Go | Version v2 obsolète | Archiver tout |
| runs/sft_v2_smoke/checkpoints/* | ~2.4 Go | Smoke test v2 | Archiver tout |
| runs/sft_v3_large/checkpoints/* sauf best | ~2.4 Go | Version v3 obsolète | Archiver tout |
| runs/sft_v6_distilled/checkpoints/* sauf best | ~2.4 Go | Version v6 obsolète | Archiver tout |
| runs/sft_benchmark | 24 Ko | Benchmark temporaire | Archiver |
| runs/sft_dryrun | 8 Ko | Dryrun temporaire | Archiver |
| runs/sft_smoke | 8 Ko | Smoke test temporaire | Archiver |
| runs/sft_v2_dryrun | 8 Ko | Dryrun v2 temporaire | Archiver |
| runs/sft_smoke_real | 966 Mo | Smoke test réel | Archiver |
| data/sft/packed_v1 | 115 Mo | Packed v1 obsolète | Archiver |
| data/sft/packed_v2 | 166 Mo | Packed v2 obsolète | Archiver |
| data/sft/packed_v3 | 706 Mo | Packed v3 obsolète | Archiver |
| data/sft/packed_v4 | 706 Mo | Packed v4 obsolète | Archiver |
| data/sft/packed_v5 | 1.2 Go | Packed v5 obsolète | Archiver |
| data/sft/packed_v6 | 1.2 Go | Packed v6 obsolète | Archiver |
| data/sft/packed | 8 Ko | Packed temporaire | Archiver |
| data/sft/packed_smoke | 12 Ko | Packed smoke temporaire | Archiver |
| data/sft/downloaded/raw/combined_v7.jsonl | 12 Go | Agrégat v7 (sources présentes) | Archiver (peut être reconstruit) |
| data/sft/downloaded/raw/combined_all.jsonl | 1.3 Go | Agrégat (sources présentes) | Archiver (peut être reconstruit) |
| data/sft/downloaded/raw/combined_with_teacher.jsonl | 1.3 Go | Agrégat avec teacher (peut être reconstruit) | Archiver |
| data/sft/downloaded/raw/combined.jsonl | 460 Mo | Agrégat (sources présentes) | Archiver (peut être reconstruit) |

## Éléments à GARDER

| Chemin | Taille | Raison |
|--------|--------|--------|
| runs/pretrain_v8/ (avec best et latest) | 2.6 Go | Run v8 actuel |
| runs/phase1_tinystories/checkpoints/best | ~300 Mo | Checkpoint phase1 |
| runs/sft_v7_massive/checkpoints/best | ~300 Mo | Meilleur modèle v7 |
| data/sft/packed_v7 | 1.2 Go | Données SFT v7 |
| data/v8/ | ~1.3 Go | Données pré-entraînement v8 |
| data/eval/ | ~100 Ko | Prompts d'évaluation |
| data/sft/downloaded/raw/sources individuels | ~14 Go | Sources brutes (nécessaires pour reconstruction) |

## Commandes d'archivage

```bash
# Créer le dossier d'archive
mkdir -p archive

# Archiver les checkpoints intermédiaires pretrain_v8
mv runs/pretrain_v8/checkpoints/step_0004500 archive/
mv runs/pretrain_v8/checkpoints/step_0004600 archive/

# Archiver les runs obsolètes (garder best pour phase1 et sft_v7)
mkdir -p archive/runs_obsoletes
mv runs/sft_v1 archive/runs_obsoletes/
mv runs/sft_v2_full archive/runs_obsoletes/
mv runs/sft_v2_smoke archive/runs_obsoletes/
mv runs/sft_v3_large archive/runs_obsoletes/
mv runs/sft_v6_distilled archive/runs_obsoletes/
mv runs/sft_benchmark archive/runs_obsoletes/
mv runs/sft_dryrun archive/runs_obsoletes/
mv runs/sft_smoke archive/runs_obsoletes/
mv runs/sft_v2_dryrun archive/runs_obsoletes/
mv runs/sft_smoke_real archive/runs_obsoletes/

# Pour phase1_tinystories, ne garder que best
mkdir -p archive/phase1_tinystories_checkpoints
mv runs/phase1_tinystories/checkpoints/step_* archive/phase1_tinystories_checkpoints/

# Archiver les packed obsolètes
mkdir -p archive/packed_obsoletes
mv data/sft/packed_v1 archive/packed_obsoletes/
mv data/sft/packed_v2 archive/packed_obsoletes/
mv data/sft/packed_v3 archive/packed_obsoletes/
mv data/sft/packed_v4 archive/packed_obsoletes/
mv data/sft/packed_v5 archive/packed_obsoletes/
mv data/sft/packed_v6 archive/packed_obsoletes/
mv data/sft/packed archive/packed_obsoletes/
mv data/sft/packed_smoke archive/packed_obsoletes/

# Archiver les agrégats combined_* (peuvent être reconstruits)
mkdir -p archive/combined_aggregates
mv data/sft/downloaded/raw/combined_v7.jsonl archive/combined_aggregates/
mv data/sft/downloaded/raw/combined_all.jsonl archive/combined_aggregates/
mv data/sft/downloaded/raw/combined_with_teacher.jsonl archive/combined_aggregates/
mv data/sft/downloaded/raw/combined.jsonl archive/combined_aggregates/
```

## Estimation de l'espace libéré

- Checkpoints pretrain_v8 intermédiaires: ~1 Go
- Runs obsolètes: ~14 Go
- Packed obsolètes: ~4 Go
- Agrégats combined_*: ~15 Go
- **Total libéré**: ~34 Go

## Espace final estimé

- runs/: ~5 Go (pretrain_v8 + phase1 best + sft_v7 best)
- data/sft/packed_v7: 1.2 Go
- data/v8/: 1.3 Go
- data/sft/downloaded/raw/sources: ~14 Go
- data/eval/: < 1 Mo
- **Total final**: ~21 Go
