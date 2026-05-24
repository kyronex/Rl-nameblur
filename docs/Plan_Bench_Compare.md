# 🗺️ Plan d'évolution post-rev 11 — ordre de complexité croissante

## 📌 Synthèse des décisions actées

| Question | Décision                                                                                                            |
| -------- | ------------------------------------------------------------------------------------------------------------------- |
| **Q1**   | Ordre de **complexité croissante** — on enchaîne du plus simple au plus complexe                                    |
| **Q2**   | **Schéma libre** — sortie ET sources JSONL modifiables sans contrainte (phase dev)                                  |
| **Q3**   | Objectif = **pertinence maximale du JSON rapport** — ventilation choisie au cas par cas selon ce qui sert l'analyse |

→ Conséquence Q3 : **pas de ventilation imposée a priori**. Chaque évolution choisit sa propre structure (par sonde / canal / domaine / thread) selon ce qui rend le rapport le plus exploitable.

---

## 🪜 Séquencement par complexité croissante

| #      | Chantier                                                   | Complexité  | Effort  | Critères impactés        | Dépendances                |
| ------ | ---------------------------------------------------------- | ----------- | ------- | ------------------------ | -------------------------- |
| **S1** | **P0-5** — Filtre §7 défensif `_enqueue()`                 | 🟢 Triviale | ~30 min | Robustesse source        | Aucune                     |
| **S2** | **C2 / P1** — Lecture `mono` + `frame_idx` dans le rapport | 🟡 Faible   | ~2 h    | #7 (2→7/10)              | Aucune                     |
| **S3** | **C1 / P1** — Ventilation fine canal `fast`                | 🟡 Moyenne  | ~2-3 h  | #3 (5→8/10), #5          | S2 recommandé (mono utile) |
| **S4** | **Backlog v2.a** — Buckets temporels (early/mid/late)      | 🟠 Moyenne+ | ~3-4 h  | #7 (7→9/10)              | S2 obligatoire             |
| **S5** | **Backlog v2.b** — Détection anomalies (spikes, drift)     | 🟠 Élevée   | ~4-5 h  | #6 (3→8/10)              | S2 + S4                    |
| **S6** | **Backlog v2.c** — Budget frame & corrélations             | 🔴 Élevée   | ~5-6 h  | #8 (4→8/10), #1 (5→8/10) | S2 + S3 + S4               |

**Logique** : chaque étape laisse le pipeline fonctionnel et le rapport exploitable. Pas d'effet tunnel. Possible d'arrêter à n'importe quelle étape.

---

## 🎯 Étape **S1 — P0-5** : démarrage immédiat

C'est l'évolution la plus simple, isolée côté source uniquement, sans impact rapport.

### Spec courte

- **Cible** : `bench/jsonl_writer.py`, méthode `_enqueue()`
- **Objectif** : refuser à l'entrée tout payload ne respectant pas la matrice §7 (champs obligatoires absents, types invalides) avant la mise en queue → évite contamination JSONL en cas de bug émetteur futur
- **Comportement** : log warning + drop (pas d'exception, le bench ne doit jamais crasher le pipeline)
- **Impact rapport** : aucun (la doc `bench-compare.md` ne change pas)

### 📄 Zones exactes à fournir pour produire le patch S1

1. **`bench/jsonl_writer.py`** — intégralité du fichier
2. **`docs/bench-jsonl-schema.md`** — section §7 uniquement (matrice champs obligatoires par canal)
3. _(optionnel — confort)_ **`bench/bench.py`** — méthodes qui appellent `_enqueue()` (pour vérifier qu'aucun appelant légitime ne produit un payload non conforme par construction)

---

## 🔄 Méthode de travail proposée

Pour chaque étape S1 → S6 :

1. **Spec courte** (5-10 lignes) — j'écris, tu valides
2. **Liste précise des zones à fournir** — je demande, tu fournis
3. **Patch + tests si applicable** — je produis
4. **Validation** — tu fais tourner, on confirme
5. **Mise à jour audit** (rev N+1) — j'acte l'évolution dans le tableau

→ Cadence courte, pas de big-bang, chaque étape clôt proprement avant la suivante.

---

## ❓ Question de démarrage

**Confirmes-tu qu'on attaque S1 maintenant ?**

Si oui → fournis les zones listées ci-dessus (`jsonl_writer.py` intégral + §7 de `bench-jsonl-schema.md`) et je produis le patch.

Si tu préfères d'abord une **vue d'ensemble plus précise** (spec détaillée de S1 → S6 avec exemples de structure JSON cible pour chacun), dis-le et je produis ce document de référence avant tout patch.
