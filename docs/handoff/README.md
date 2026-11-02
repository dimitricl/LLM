# Handoff — reprise sur le Mac Mini

Ce dossier permet de reprendre la conversation OpenCode sur une autre machine
sans perdre le contexte du cadrage.

## État du projet au 2026-11-02

- Spec validée : `docs/specs/2026-11-02-tiny-llm-mlx-design.md` (commit `8c7bf30`).
- Plan d'implémentation validé : `docs/plans/2026-11-02-tiny-llm-mlx.md`
  (commit `e44bc2f`, 21 tâches, 137 étapes).
- **Aucun code n'est encore écrit.** La prochaine action est la Tâche 0
  (bootstrap : `pyproject.toml`, `.gitignore`, `src/tinyllm/__init__.py`,
  `tests/test_import.py`).
- Mode d'exécution pas encore choisi (subagent-driven recommandé).

## Reprendre la conversation

```bash
# 1. Récupérer le dépôt sur le Mac Mini
git clone https://github.com/dimitricl/LLM.git
cd LLM

# 2. Restaurer la session OpenCode
gunzip -c docs/handoff/2026-11-02-session.json.gz > /tmp/tinyllm-session.json
opencode session import /tmp/tinyllm-session.json

# 3. Lister les sessions et reprendre celle importée
opencode session list
opencode -s <session-id>
```

Notes :

- L'export a été produit avec la version d'OpenCode installée sur le MacBook.
  Si le format a changé, `opencode session import` le signalera ; dans ce cas,
  repartir d'une session neuve en demandant simplement :
  « Lis `docs/HANDOFF` (ce fichier) et `docs/plans/2026-11-02-tiny-llm-mlx.md`,
  puis exécute la Tâche 0. »
- Le chemin du projet dans la session importée peut être celui du MacBook. Si
  OpenCode refuse d'opérer dans le nouveau dossier, demander à l'agent de
  déplacer la session vers le répertoire courant (outil `session_move`), ou
  repartir d'une session neuve comme ci-dessus.
- Le fichier `.json.gz` contient la transcription complète (messages et
  sorties d'outils). Il ne contient pas de secrets.
