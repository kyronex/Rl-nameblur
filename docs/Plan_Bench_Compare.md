# 📊 Score de pertinence par critère — ventilation par flux + commentaires d'audit

> **En-tête de version**
>
> - `audit_revision`: 9
> - `date`: 2026-05-19
> - `contrat_référence`: L0.4 (`schema_version=1`, figé)
> - `périmètre`: sources JSONL (`bench/jsonl_writer.py` + `bench/bench.py`) + traitement (`logs/bench_compare.py`) + inventaire sondes (`docs/bench-probes.md`, 74 sondes / 10 domaines)
> - `nouveauté rev 9`: **externalisation historique** — révisions, incertitudes résolues, bugs clos, observations résolues déplacés vers [`audit-bench-historique.md`](./audit-bench-historique.md). Aucun changement d'état sur les items actifs.

---

## Légende

- **Source ok/ko** = qualité de la donnée émise dans le JSONL **au regard du contrat L0.4**
- **Traitement ok/ko** = qualité de l'exploitation par `bench_compare.py`
- 🟢 / 🟡 / 🔴 = état (ok / partiel ou ambigu / ko ou violation)
- 🆕 = critère ajouté ou réévalué
- 🔄 = origine du manque reclassifiée
- ✅ = statut vérifié par audit code ou lecture normative
- **N/A** = non applicable
- **RAS** = rien à signaler

---

## Tableau de synthèse

| Id  | Critère                                       | `bench_fast.jsonl`    | `bench_agg.jsonl`     | `bench_frame.jsonl`           | `<session>.json` | Origine du manque                                     |
| --- | --------------------------------------------- | --------------------- | --------------------- | ----------------------------- | ---------------- | ----------------------------------------------------- |
| 1   | **Identification goulot principal**           | ✅ Source ok          | ✅ Source ok          | ✅ Source ok                  | 🟡 5/10          | Traitement                                            |
| 2   | **Statistiques robustes (percentiles)**       | ✅ Source ok          | ✅ Source ok          | ✅ Source ok                  | 🟢 8/10          | RAS — `method='inclusive'` confirmé                   |
| 3   | **Couverture canal fast**                     | 🟢 ✅ Source ok       | N/A                   | N/A                           | 🟡 5/10          | Traitement partiel — voir C1                          |
| 4   | **Couverture canal agg**                      | N/A                   | 🟢 ✅ Source ok       | N/A                           | 🟢 9/10          | RAS                                                   |
| 5   | **Couverture canal frame**                    | N/A                   | N/A                   | 🟢 ✅ Source ok               | 🟡 6/10          | Traitement (limite v1 documentée)                     |
| 6   | **Détection anomalies**                       | ✅ Source ok          | ✅ Source ok          | ✅ Source ok                  | 🔴 0/10          | Traitement (hors scope v1)                            |
| 7   | **Analyse temporelle / drift**                | 🟢 ✅ Source ok       | 🟢 ✅ Source ok       | 🟢 ✅ Source ok               | 🔴 2/10          | Traitement — `mono` ignoré                            |
| 8   | **Corrélations inter-probes**                 | 🟡 Source partielle   | 🟡 Source partielle   | 🟢 ✅ Source ok               | 🔴 1/10          | Traitement (hors scope v1)                            |
| 9   | **Cohérence count vs samples**                | ✅ Source ok          | ✅ Source ok          | ✅ Source ok                  | 🟡 4/10          | Traitement — voir C5                                  |
| 10  | **Sémantique cumulative vs fenêtrée**         | 🟢 ✅ Source ok       | 🟢 ✅ Source ok       | 🟡 ✅ gauges instantanés §5.2 | 🟡 5/10          | Source documentée + Traitement                        |
| 11  | **Classification probes/rates/gauges/counts** | ✅ Source ok          | ✅ Source ok          | ✅ Source ok                  | 🟡 5/10          | Traitement — `counts` frame ignoré (C2)               |
| 12  | **Conformité schéma documenté**               | 🟢 ✅ Source ok       | 🟢 ✅ Source ok       | 🟢 ✅ Source ok               | 🟡 5/10          | Traitement — ingestion permissive (C3)                |
| 13  | **Budget frame / actionnabilité**             | ✅ Source ok          | ✅ Source ok          | ✅ Source ok                  | 🔴 2/10          | Traitement                                            |
| 14  | **Métadonnées session**                       | 🟢 ✅ Source ok (5/5) | 🟢 ✅ Source ok (5/5) | 🟢 ✅ Source ok (5/5)         | 🔴 1/10          | Traitement — méta jamais lues                         |
| 15  | **Validation `schema_version`**               | 🟢 ✅ émis = 1 (§2.1) | 🟢 ✅ émis = 1 (§2.1) | 🟢 ✅ émis = 1 (§2.1)         | 🔴 0/10          | Traitement — confirmé absent (C3)                     |
| 16  | **Respect matrice §7 sections/canal**         | 🟢 ✅ vérifié §7      | 🟢 ✅ vérifié §7      | 🟢 ✅ vérifié §7              | 🔴 2/10          | Traitement — aucune validation (C3)                   |
| 17  | **Inventaire sondes exhaustif**               | 🟢 ✅ 15 sondes fast  | 🟢 ✅ multi-domaines  | 🟢 ✅ multi-domaines          | 🟡 4/10          | Traitement — `motion`/`associator` non valorisés (C4) |

---

## 🔬 Constats actifs `bench_compare.py` (audit rev 8)

| Id  | Constat                                                                                         | Sévérité |
| --- | ----------------------------------------------------------------------------------------------- | -------- |
| C1  | Détection sondes `fast_*` par préfixe (`startswith`) — non extensible aux familles dynamiques   | 🟡       |
| C2  | `counts` frame §5.2 non traités séparément dans `_agg_probes` (risque latent)                   | 🟡       |
| C3  | Aucune lecture de `schema_version` / `session_id` (champ) / `mode` / `mono`                     | 🔴 P0    |
| C4  | Sondes `motion_*` / `associator_*` / `selector_source_*` ingérées génériquement, non valorisées | 🟢       |
| C5  | Pas de croisement `samples_exact` (frame) vs `count` cumulé (agg)                               | 🟡       |

> Architecture pipeline et conformité fonctionnelle vérifiée — voir [annexe §audit rev 8](./audit-bench-historique.md#audit-code-bench_comparepy-rev-8).

---

## 🎯 Checklist d'audit priorisée

### 🔥 P0 — Bugs ouverts

| Id   | Description                                                        | Origine    | Statut         |
| ---- | ------------------------------------------------------------------ | ---------- | -------------- |
| P0-4 | `schema_version` non validé — §2.2 fallback absent                 | Traitement | 🔴 Ouvert (C3) |
| P0-5 | Matrice §7 non vérifiée à l'ingestion — `_agg_probes` accepte tout | Traitement | 🔴 Ouvert (C3) |

### 🟡 P1 — Dette traitement

1. Sondes `fast_*` détectées par préfixe — famille non extensible (C1).
2. `selector_source_<name>` ingérée comme probe standard — pas d'agrégation par pattern (O17).
3. Métadonnées session non exposées dans `<session>.json` (C3).
4. `counts` frame §5.2 non traités explicitement (C2).
5. `mono` ignoré — aucune analyse temporelle bucketisée.
6. `associator_hungarian_rejected_total` agrégat non désagrégeable (O18).

### 🟢 P2 — Améliorations traitement (backlog v2)

1. Buckets temporels via `mono` (§2.1 garantit le champ).
2. Détection d'anomalies par règles simples.
3. Budget frame + `budget_consumed_pct`.
4. Exploitation complète canal `frame` (corrélations, timeline).

### 📋 P3 — Documentation

1. Contrat émetteur ↔ `bench_compare.py` — référencer §7 + inventaire (10 domaines, 74 sondes).

---

## Verdict d'audit

- **Sources** : qualité **~99%**. Dette résiduelle : O2 (filtre §7 défensif absent dans `_enqueue()`).
- **Traitement** : qualité **~48%**. Manques restants : validation de schéma (P0-4/5) + exploitation métadonnées (P1).

### Incertitudes actives (Z\*)

| Id  | Description                | Document nécessaire   | Priorité |
| --- | -------------------------- | --------------------- | -------- |
| Z6  | Couverture tests existante | Arborescence `tests/` | 🟢 P2    |

→ **Prochaine session recommandée** : implémentation P0-4 + P0-5 (validation `schema_version` + filtre §7 défensif à l'ingestion).

---

> 🔒 **Rappel §8** : toute modification du schéma requiert incrément `schema_version` + ticket dédié. Aucun patch silencieux côté source.

---

## 📎 Annexe — Observations code actives

| Id  | Fichier            | Sévérité       | Description                                                                |
| --- | ------------------ | -------------- | -------------------------------------------------------------------------- |
| O2  | `jsonl_writer.py`  | 🟡 dette       | Pas de filtre §7 défensif dans `_enqueue()`                                |
| O12 | `bench.py`         | 🟡 mineur      | Buffer `_frame_probes` accumule avant filtrage                             |
| O14 | `bench.py`         | 🟢 négligeable | Race condition théorique `_maybe_start_writers`                            |
| O15 | `bench.py`         | 🟡 mineur      | `reset()` ne stoppe pas les writers                                        |
| O16 | `bench-probes.md`  | 🟡 info        | Domaine `associator` — 8 sondes ingérées génériquement (C4)                |
| O17 | `bench-probes.md`  | 🟢 info        | `selector_source_<name>` famille dynamique — pas d'agrégation par pattern  |
| O18 | `bench-probes.md`  | 🟡 info        | `associator_hungarian_rejected_total` agrège 2 causes non désagrégeables   |
| O19 | `bench-probes.md`  | 🟢 info        | `tracker_*` (runtime) et `registry_*` (post-session) coexistent par design |
| C1  | `bench_compare.py` | 🟡 info        | Détection `fast_*` par préfixe — non extensible                            |
| C3  | `bench_compare.py` | 🔴 P0          | Aucune lecture méta-champs §2.1                                            |
| C5  | `bench_compare.py` | 🟡 info        | Pas de croisement `samples_exact` vs `count` cumulé                        |

---

## 📚 Historique & éléments clos

Externalisés vers [`audit-bench-historique.md`](./audit-bench-historique.md) :

- **Historique des révisions** (rev 1 → rev 9)
- **Incertitudes résolues** (Z1, Z2, Z3, Z4, Z5)
- **Bugs résolus** (P0-1, P0-2, P0-3, O13)
- **Audit code détaillé `bench_compare.py`** (rev 8, architecture pipeline, conformité fonctionnelle vérifiée)

---

## 📝 Création annexe — `audit-bench-historique.md`

> Fichier compagnon à créer en parallèle, contenant :
>
> 1. Tableau **Historique des révisions** (8 lignes, rev 1 → rev 8) + entrée rev 9 (« externalisation historique »).
> 2. Tableau **Incertitudes résolues** (Z1, Z2, Z3, Z4, Z5) avec révision de clôture et résolution.
> 3. Tableau **Bugs résolus** (P0-1, P0-2, P0-3, O13) avec révision de clôture et résolution.
> 4. Section **Audit code `bench_compare.py` rev 8** : schéma architecture pipeline + checklist conformité fonctionnelle vérifiée (percentiles `inclusive`, seuil 20, mode single, déplacement atomique, gestion `OSError`, etc.).
>
> Le contenu intégral de ces sections est repris **tel quel** depuis la rev 8 du présent document, sans modification de fond.
