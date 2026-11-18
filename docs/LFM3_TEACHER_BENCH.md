# LFM3 1.7B — peut-il servir de professeur pour TinyLLM ?

Bench réalisé le 2026-11-12 sur `lfm3:1.7b` (Ollama, 1,7 Go, Q4).
Reproductible avec `scripts/bench_teacher.py --model lfm3:1.7b`.

## Réglage indispensable

LFM3 est un modèle de raisonnement. Sans `"think": false`, il consomme tout
`num_predict` dans le champ `thinking` et renvoie un `content` vide :

```json
{"message": {"content": "", "thinking": "The user asks: ..."}}
```

Avec `think: false`, le champ `thinking` reste vide et `content` est rempli.
`format: "json"` renvoie une erreur 501 sur cette version d'Ollama : pas de
contrainte grammaticale, il faut parser et réparer côté client.

## Résultats

| Mesure | LFM3 1.7B | TinyLLM `anneal2` |
|---|---:|---:|
| Batterie conversation (16 cas) | 8/16 | 15/16 |
| Fuites small talk (10 questions) | **0/10** | 3/10 |
| Débit | 52 tok/s | ~250 tok/s |
| JSON valide (lots de 5 conversations) | 3/5 | — |

Les 8 échecs de la batterie conversation sont attendus et peu informatifs : le
modèle répond « je suis un LFM, créé par Liquid AI », dépasse 40 mots, utilise
du markdown et des emojis. Ce n'est pas un défaut de raisonnement, c'est un
désaccord d'identité et de style. En revanche, sur les questions générales, il
répond juste là où TinyLLM déraille : Paris, 56, photosynthèse, cuisson du riz,
poème cohérent, 0 fuite.

## Fiabilité factuelle : le point bloquant

Sur une demande de 10 paires question/réponse « connaissances générales », il
en a produit 14 (consigne non respectée), et parmi elles :

| Réponse produite | Problème |
|---|---|
| « Combien de continents ? » → « Six » | contestable |
| « Les étoiles du soir ? » → « Le coucher du soleil » | faux (Vénus) |
| « Vitesse max d'un avion de ligne ? » → « Mach 900 » | absurde |
| « Le café français ? » → « Un thé brûlé avec sucre et crème » | absurde |
| « Le Petit Prince ? » → « 99 pages » | confond livres et pages |
| « Plus long voyage à pied ? » → « New York–Los Angeles » | faux |

Conclusion : la fluidité n'est pas la connaissance. Utilisé sans filtre pour
générer des faits, il injecte des erreurs dans TinyLLM — exactement le problème
qu'on cherche à corriger.

## JSON et discipline de format

- 3 lots sur 5 strictement valides (15 conversations en 22 s, ~1 conversation
  toutes les 1,5 s).
- Erreurs typiques : un `]` en trop, `"role": "assistance"`, virgule manquante,
  parfois un raisonnement en clair après le JSON (« Wait, need three
  variants… »).
- Une réparation naïve (retirer le crochet final) récupère une partie des lots,
  pas tous. Il faut valider conversation par conversation et régénérer.

Comparaison rapide, même prompt :

| Modèle | JSON valide (10 Q/R) | Remarque |
|---|---|---|
| `lfm3:1.7b` | non (14 items, virgules manquantes) | 4 erreurs factuelles |
| `qwen3:4b-instruct` | 0/5 lots | plus lent, même problème de crochets |
| `qwen3:8b` | non (virgules finales) | 35 s |
| `gemma3:4b` | **oui, 10 items** | 1 erreur factuelle |

Sur cet échantillon unique, `gemma3:4b` est le meilleur générateur JSON, et
`lfm3` le plus rapide.

## Dérive de langue

À `temperature 0.3` avec des prompts courts, 2 réponses sur 16 sont parties en
portugais (« É daqui, tudo bem! ») et en espagnol (« Soy un modelo… »). À
`temperature 0.7`, 24/24 réponses sont restées en français. La dérive est rare
mais réelle : un filtre de langue est obligatoire si on génère en masse.

## Verdict

**Non** comme professeur de conversation : il apprendrait à TinyLLM à être
verbeux, à écrire en markdown et à s'appeler LFM.

**Non** comme source de faits non vérifiés : trop d'erreurs.

**Oui** pour un usage étroit et sûr : **paraphraser des données déjà
vérifiées**. On garde les réponses assistant écrites à la main, on ne fait
varier que les prompts utilisateur. Aucune connaissance nouvelle n'est
introduite, et ça attaque un vrai problème du dataset actuel (les variantes de
frappe sont générées par règles, donc artificielles).

## Piste concrète

1. Extraire les prompts utilisateur des 2 536 conversations d'annealing.
2. Demander à LFM3 3 reformulations naturelles par prompt (temperature 0,5).
3. Valider : langue française, longueur proche, pas de markdown, pas de
   « LFM »/« Liquid ».
4. Réinjecter les variantes avec les réponses assistant d'origine.
5. Reconstruire `packed_v8_anneal4` et refaire une passe courte.

Coût estimé : ~1,5 s par prompt, soit ~1 h pour 2 500 prompts, avant filtrage.
