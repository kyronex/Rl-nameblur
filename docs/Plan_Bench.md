# 📋 Plan de mise en œuvre séquencé — Post-audit B-04b (v4)

> **Objectif** : passer du backlog audité (60 sondes différées + 3 décisions transverses + 3 dettes legacy + 1 arbitrage structurel) à un système instrumenté, propre et prêt pour B-04c.
>
> **Principe directeur** : séquentialité stricte, chaque lot indépendant testable, rollback possible à chaque borne. Aucun lot ne démarre tant que le précédent n'est pas validé.
>
> **Arbitrages v3 → v4** :
>
> - **Option C de sortie JSONL pleinement spécifiée** : 2 fichiers (`bench_frame.jsonl` + `bench_agg.jsonl`), lignes structurées **par domaine** via préfixe de nommage T-2.
> - Schéma JSON figé pour L0 (cf. § L0.4).
> - `agg_interval` exposé en config (défaut : 1.0 s).

---

## 🎯 Vue d'ensemble

| Lot    | Scope                                                               | Effort  | Précondition           |
| ------ | ------------------------------------------------------------------- | ------- | ---------------------- |
| **L0** | Fondations transverses (API bench + schéma JSON + migration config) | ~2 h 30 | Audit B-04b clôturé ✅ |
| **L1** | Arbitrage structurel : `created_ts` natif sur `Mask`                | ~30 min | L0                     |
| **L2** | Purge dettes legacy (3 blocs + arbitrage `tracker.stats()`)         | ~1 h    | L1                     |
| **L3** | Déploiement 60 sondes différées (par fichier)                       | ~3 h    | L2                     |
| **L4** | Validation intégration (smoke test 60 s)                            | ~30 min | L3                     |
| **L5** | Livraison B-04b : récap consolidé + bascule B-04c                   | ~30 min | L4                     |

**Effort total estimé** : ~8 h, étalable sur 2 sessions.

---

## 🔵 Lot L0 — Fondations transverses

> ⚠️ **Lot critique** : modifie l'API `bench`, la structure config et fige le schéma JSONL. Tout ce qui suit dépend de ces fondations. À traiter en premier, en un seul commit atomique.

### L0.1 — API `bench` : `snapshot_all()` + `snapshot_frame()`

- **Fichier** : `bench.py`
- **Actions** :
  - `snapshot_frame() -> dict[str, dict]` : snapshot **par frame** destiné à `bench_frame.jsonl`. Structure groupée par domaine.
  - `snapshot_all() -> dict[str, dict]` : snapshot **glissant cumulatif** depuis start, destiné à `bench_agg.jsonl`. Structure groupée par domaine.
- **Contrat** :
  - Thread-safe (lock interne).
  - Idempotent sur snapshot vide (retourne `{}` ou domaines vides explicitement).
  - Pas de reset implicite — la sémantique cumulative est explicite.
  - Groupement par domaine dérivé du **préfixe de nommage T-2** (`<domaine>_<action>[_<qualifieur>]` → domaine = premier segment).
- **Validation** : tests unitaires (snapshot vide, snapshot non-vide, concurrence basique, cohérence groupement par domaine).

### L0.2 — Décisions transverses actées

- **T-1 (Option C)** : 2 fichiers JSONL en sortie bench :
  - `bench_frame.jsonl` (1 ligne / frame).
  - `bench_agg.jsonl` (1 ligne / `agg_interval`).
- **T-2 (nommage sondes)** : convention verrouillée `<domaine>_<action>[_<qualifieur>]`. Cette convention **conditionne le groupement JSON** (cf. L0.4).
- **T-3 (rétention)** : cumulatif depuis start pour `snapshot_all()` ; snapshot frame indépendant pour `snapshot_frame()`.
- **Fichiers touchés** : `bench.py`, `main.py` (consommateur des deux snapshots).

### L0.3 — Migration config `debug.csv.*` → `debug.bench.jsonl.*`

> 📦 **Décision actée n°3** : alignement de la nomenclature config avec l'Option C.

- **Fichier** : `config.yaml`
- **Avant** : `debug.csv.*` (legacy mono-fichier).
- **Après** :

  ```yaml
  debug:
    bench:
      jsonl:
        per_frame:
          enabled: true
          path: "logs/bench_frame.jsonl"
        aggregated:
          enabled: true
          path: "logs/bench_agg.jsonl"
          agg_interval: 1.0 # secondes
  ```

- **Validation** : démarrage application sans crash, fichiers créés au chemin attendu, hot-reload du chemin testé.

### L0.4 — Schéma JSONL figé (Option C)

> 🔒 **Schéma normatif** : toute évolution ultérieure passe par un nouveau ticket.

**`bench_frame.jsonl`** — 1 ligne / frame :

```json
{
  "ts": 1234567.890,
  "frame_id": 42,
  "main":     { "loop_total_ms": 12.3, "frame_dropped": 0, "snapshot_build_ms": 0.4, ... },
  "tracker":  { "apply_fast_direct_ms": 0.8, "fast_applied": 3, "match_score_p50": 0.72, ... },
  "registry": { "masks_confirmed": 4, "masks_pending": 1, "masks_lost": 0, ... },
  "motion":   { "velocity_magnitude": 15.2, "dead_zone_hit": 0, ... },
  "detect":   { "detect_pipeline_ms": 5.2, ... },
  "boxes":    { "boxes_pipeline_ms": 4.1, "boxes_n_final": 2, "boxes_n_raw": 8, ... },
  "fast":     { "fast_ncc_score": 0.81, "fast_hot_reload": 0, ... },
  "mask":     { "mask_age_s": 1.4, ... }
}
```

**`bench_agg.jsonl`** — 1 ligne / `agg_interval` :

```json
{
  "ts": 1234568.000,
  "window_s": 1.0,
  "main":     { "loop_total_ms_p50": 11.8, "loop_total_ms_p95": 18.2, "frames_total": 60, ... },
  "tracker":  { "fast_applied_total": 87, "match_score_p50": 0.72, "match_score_p95": 0.91, ... },
  "registry": { "mask_lifetime_s_p50": 1.4, "mask_created_rate": 3.1, ... },
  "motion":   { ... },
  "detect":   { ... },
  "boxes":    { ... },
  "fast":     { ... },
  "mask":     { ... }
}
```

**Règles de groupement** :

| Préfixe sonde                                             | Domaine JSON |
| --------------------------------------------------------- | ------------ |
| `loop_*`, `frame_*`, `snapshot_*`, `give_frame_*`         | `main`       |
| `apply_*`, `match_*`, `fast_applied`, `tracker_*`         | `tracker`    |
| `masks_*`, `mask_lifetime_*`, `mask_created_*`            | `registry`   |
| `velocity_*`, `dead_zone_*`, `predicted_*`, `staleness_*` | `motion`     |
| `detect_*`                                                | `detect`     |
| `boxes_*`                                                 | `boxes`      |
| `fast_*`                                                  | `fast`       |
| `mask_age_*`, `mask_*` (instance)                         | `mask`       |

**Domaines vides** : omis de la ligne JSON (pas de clé `"motion": {}` vide).

### 🔒 Borne L0

- ✅ `snapshot_frame()` + `snapshot_all()` livrés, testés unitairement.
- ✅ `config.yaml` migré, anciens chemins `debug.csv.*` supprimés.
- ✅ Schéma L0.4 figé et documenté dans `bench.py` (docstring module).
- ✅ Smoke run 10 s : 2 fichiers créés, lignes parsables, groupement par domaine correct.
- ✅ Tag git `b04b-L0`.

---

## 🟣 Lot L1 — Arbitrage structurel : `created_ts` natif

- **Fichier** : `core/mask.py`
- **Action** : ajout champ `created_ts: float` dans le dataclass `Mask`, initialisé à la création (Option A actée en audit).
- **Conséquence** : `registry.py` consomme `mask.created_ts` au lieu de structures parallèles (purge associée traitée en L2).
- **Validation** : test unitaire création `Mask`, `created_ts` cohérent avec `time.monotonic()`.

### 🔒 Borne L1

- ✅ `Mask.created_ts` natif.
- ✅ Consommateurs migrés.
- ✅ Tag git `b04b-L1`.

---

## 🟠 Lot L2 — Purge dettes legacy

### L2.1 — Purge `_motion_stats` / `get_and_reset_stats()`

- **Fichier** : `motion.py`
- **Action** : suppression intégrale du dict `_motion_stats`, de `get_and_reset_stats()` et de tous ses imports/appels (notamment `main.py`).
- **Précondition** : sondes motion (L3.5) seront posées via `bench` direct → pas de double instrumentation.

### L2.2 — Purge `frame_count` / `fps_timer` legacy

- **Fichier** : `main.py`
- **Action** : suppression du compteur FPS legacy, remplacement par lecture `bench` (sonde `loop_total` + agrégat).

### L2.3 — Arbitrage `tracker.stats()`

- **Fichier** : `tracker.py`
- **Action différée au moment de L2** : audit ciblé des consommateurs `tracker.stats()` :
  - Si aucun consommateur productif → **suppression**.
  - Sinon → **scopé en P-04** (optim ultérieure), tracé explicitement.
- **Livrable obligatoire** : décision écrite dans le commit + Plan_Tracker.md.

### L2.4 — Purge bloc `_b01_stats` dans `registry.py`

- **Fichier** : `tracker/registry.py`
- **Action** : suppression du dict, compteurs, logs, imports, commentaires associés à `_b01_stats` (legacy B-01).

### 🔒 Borne L2

- ✅ `_motion_stats` et `get_and_reset_stats()` supprimés.
- ✅ `frame_count`/`fps_timer` legacy retirés de `main.py`.
- ✅ Décision `tracker.stats()` tracée explicitement (supprimé OU scopé P-04).
- ✅ Bloc `_b01_stats` intégralement purgé de `registry.py`.
- ✅ `main.py` lit exclusivement via `bench`.
- ✅ Smoke test : HUD FPS toujours cohérent.
- ✅ Tag git `b04b-L2`.

---

## 🔴 Lot L3 — Déploiement des 60 sondes différées

> 📐 **Stratégie** : déploiement organisé **par fichier** (9 sous-lots logiques L3.1 → L3.9). La granularité des commits est laissée à l'équipe.
>
> **Contrainte non-négociable** : l'intégralité de L3 doit être livrée avant la borne L4.
>
> **Ordre recommandé** : **L3.8 en premier** (lève 3 suspensions : L3.3, L3.4, L3.6).
>
> **Légende statuts** :
>
> - 🔒 Différé : à poser pendant L3.
> - 🟡 Suspendu : conditionné à l'audit L3.8.
> - 🟢 Statu quo : conservé, aucune action.
> - ❌ Écarté : non retenu (motif tracé).

---

### L3.1 — `main.py` (5 sondes)

| Sonde             | Type  | Emplacement                                                               | Domaine JSON |
| ----------------- | ----- | ------------------------------------------------------------------------- | ------------ |
| `frame_dropped`   | count | §1 — branche `if frame is None` avant `continue`                          | `main`       |
| `give_frame_slow` | count | §2 — après `detector.give_frame(...)`                                     | `main`       |
| `give_frame_fast` | count | §2 — après `fast_tracker.give_frame_and_views(...)` (si snapshot)         | `main`       |
| `snapshot_build`  | timer | §2 — englobant `snapshot = [...]` + `views = [...]`                       | `main`       |
| `loop_total`      | timer | Englobant l'itération complète `while True:` (de `maybe_reload` à fin §8) | `main`       |

**Existant préservé** : 9 sondes conformes au plan.

**Dettes legacy purgées** : traitées en L2.2.

---

### L3.2 — `tracker/registry.py` (7 sondes)

> **Précondition dure** : L1 (`Mask.created_ts` natif) + L2.4 (purge `_b01_stats`).

| Sonde                   | Type  | Emplacement                                                       | Domaine JSON |
| ----------------------- | ----- | ----------------------------------------------------------------- | ------------ |
| `masks_confirmed`       | probe | Fin `tick()` — `len([m for m in masks if m.state==CONFIRMED])`    | `registry`   |
| `masks_pending`         | probe | Fin `tick()` — équivalent PENDING                                 | `registry`   |
| `masks_lost`            | probe | Fin `tick()` — équivalent LOST                                    | `registry`   |
| `mask_created_rate`     | count | Branche création nouveau `Mask`                                   | `registry`   |
| `mask_expired_rate`     | count | Branche suppression mask (TTL expiré)                             | `registry`   |
| `mask_lifetime_s`       | probe | À la suppression — `now - mask.created_ts`                        | `registry`   |
| `mask_state_transition` | count | Toute transition d'état (PENDING→CONFIRMED, CONFIRMED→LOST, etc.) | `registry`   |

**Statu quo 🟢** : sondes existantes conformes (cf. récap audit étape 3).

---

### L3.3 — `tracker/tracker.py` (8 sondes)

> **Suspension levée par L3.8** : les branches `apply_detections_fast` deviennent instrumentables une fois l'audit `FastTrackThread` consolidé.

| Sonde                         | Type  | Emplacement                              | Domaine JSON |
| ----------------------------- | ----- | ---------------------------------------- | ------------ |
| `apply_detections_slow`       | timer | Englobant la méthode                     | `tracker`    |
| `apply_detections_fast`       | timer | Englobant la méthode (🟡 levé par L3.8)  | `tracker`    |
| `match_score_slow`            | probe | Boucle d'association slow — score retenu | `tracker`    |
| `match_score_fast`            | probe | Boucle d'association fast — score retenu | `tracker`    |
| `tracker_unmatched_dets_slow` | count | Détections slow non associées            | `tracker`    |
| `tracker_unmatched_dets_fast` | count | Détections fast non associées            | `tracker`    |
| `tracker_new_mask_from_slow`  | count | Création mask depuis détection slow      | `tracker`    |
| `tracker_new_mask_from_fast`  | count | Création mask depuis détection fast      | `tracker`    |

---

### L3.4 — `tracker/associator.py` (8 sondes)

> **Suspension levée par L3.8** : les branches `source="fast"` deviennent instrumentables.

| Sonde                                      | Type  | Emplacement                                 | Domaine JSON |
| ------------------------------------------ | ----- | ------------------------------------------- | ------------ |
| `associator_iou_score`                     | probe | Calcul IoU par paire candidate              | `tracker`    |
| `associator_dist_score`                    | probe | Calcul distance par paire candidate         | `tracker`    |
| `associator_final_score_slow`              | probe | Score final source=slow                     | `tracker`    |
| `associator_final_score_fast`              | probe | Score final source=fast (🟡 levé par L3.8)  | `tracker`    |
| `associator_rejected_below_threshold_slow` | count | Paire rejetée seuil slow                    | `tracker`    |
| `associator_rejected_below_threshold_fast` | count | Paire rejetée seuil fast (🟡 levé par L3.8) | `tracker`    |
| `associator_pairs_evaluated`               | count | Nombre paires évaluées par tick             | `tracker`    |
| `associator_pairs_accepted`                | count | Nombre paires acceptées par tick            | `tracker`    |

---

### L3.5 — `tracker/motion.py` (5 sondes)

> **Précondition dure** : L2.1 (purge `_motion_stats`) livré.

| Sonde                      | Type  | Emplacement                                                     | Domaine JSON |
| -------------------------- | ----- | --------------------------------------------------------------- | ------------ |
| `dead_zone_hit`            | count | `apply_detection()` — branche dead zone position (1ère branche) | `motion`     |
| `velocity_reset_dt_slow`   | count | `apply_detection()` — branche `dt > config.dt_slow_max`         | `motion`     |
| `velocity_reset_teleport`  | count | `apply_detection()` — branche `dist > config.teleport_thresh`   | `motion`     |
| `velocity_magnitude`       | probe | `apply_detection()` post-clamp source=="slow", `sqrt(vx²+vy²)`  | `motion`     |
| `predicted_clamped_screen` | count | `predict_position()` — si `x` ou `y` clampé aux bornes écran    | `motion`     |

**Statu quo 🟢** : Logger `"motion"` inutilisé, commentaire `# Sondes bench (nouvelles)` — conservés.

**Écartées ❌** : `apply_detection_duration` (doublon), `velocity_clamped_x/y/w/h` (non prioritaire), `predict_position_calls` (déductible), `damping_factor` (violation site unique), `dt_predict_ms` (doublon `staleness_slow_ms`), `predict_cold_start` (low value).

**Existant** : 2 sondes B-04b conformes (`staleness_slow_ms`, `staleness_capped`).

---

### L3.6 — `detection/detect.py` (3 sondes)

> **Arbitrage `ncc_score` tranché par L3.8** : sonde portée par `fast_track_thread.py`, pas dupliquée ici.

| Sonde                 | Type  | Emplacement                            | Domaine JSON |
| --------------------- | ----- | -------------------------------------- | ------------ |
| `detect_pipeline`     | timer | Englobant `detect_plates()`            | `detect`     |
| `detect_n_plates_in`  | probe | Entrée fonction — `len(plates_input)`  | `detect`     |
| `detect_n_plates_out` | probe | Sortie fonction — `len(plates_output)` | `detect`     |

---

### L3.7 — `detection/boxes.py` — `process_channel` (10 sondes)

| Sonde                         | Type  | Emplacement                                           | Domaine JSON |
| ----------------------------- | ----- | ----------------------------------------------------- | ------------ |
| `boxes_pipeline`              | timer | Englobant tout `process_channel`                      | `boxes`      |
| `boxes_validate_bg`           | timer | Autour de `_validate_background`                      | `boxes`      |
| `boxes_validate_text`         | timer | Autour de `_validate_text`                            | `boxes`      |
| `boxes_n_raw`                 | probe | Après `_extract_raw_boxes` — `len(boxes)`             | `boxes`      |
| `boxes_n_after_geometry`      | probe | Après `_filter_geometry` — `len(geometryed)`          | `boxes`      |
| `boxes_n_after_validate_bg`   | probe | Après `_validate_background` — `len(validated_b)`     | `boxes`      |
| `boxes_n_after_validate_text` | probe | Après `_validate_text` — `len(validated_t)`           | `boxes`      |
| `boxes_n_after_bands`         | probe | Après `_filter_horizontal_bands` — `len(banded)`      | `boxes`      |
| `boxes_n_after_alignment`     | probe | Après `_filter_horizontal_alignment` — `len(aligned)` | `boxes`      |
| `boxes_n_final`               | probe | Après `_filter_perspective_gradient` — `len(plates)`  | `boxes`      |

**Écartées ❌** : `boxes_extract_raw` / `boxes_adjust_resolve_1` / `boxes_split_wide` / `boxes_filter_bands` / `boxes_filter_alignment` / `boxes_filter_perspective` (couverts par `boxes_pipeline` + probes), `boxes_n_after_split` / `boxes_n_after_merge` (peu discriminants).

**Existant** : 0 sonde.

---

### L3.8 — `threads/fast_track_thread.py` (8 sondes) — **PIVOT**

> 🔑 **Sous-lot prioritaire** : son audit lève les suspensions L3.3 (`apply_detections_fast`) et L3.4 (branches `source="fast"` dans associator), et tranche le doublon `ncc_score` L3.6.

- 8 sondes nouvelles + suppression sonde `fast_margin` (validée pour suppression en audit étape 9).
- Inclut `fast_ncc_score` (résolution arbitrage transverse `ncc_score`).
- Inclut `fast_hot_reload` (count) — alimente vérification L4.3.

> Détail exhaustif des 8 sondes à reprendre depuis le récap d'audit étape 9 (`fast_track_thread.py`).

**Domaine JSON** : `fast`.

---

### L3.9 — `core/mask.py` (6 sondes)

> **Précondition dure** : L1 (`created_ts` natif).
> Détail exhaustif des 6 sondes à reprendre depuis le récap d'audit étape 10 (`core/mask.py`).

**Domaine JSON** : `mask`.

**Existant** : 0 sonde (dataclass pur, conforme).

---

### 🔒 Borne L3

- ✅ 60 sondes nouvelles actives (5 + 7 + 8 + 8 + 5 + 3 + 10 + 8 + 6 = **60**).
- ✅ 26 sondes existantes préservées.
- ✅ 1 sonde supprimée (`fast_margin`).
- ✅ Bench JSONL contient les 85 sondes attendues sur session courte, **réparties dans les bons domaines JSON** (cf. L0.4).
- ✅ Aucune régression FPS (delta < 5% — validé en L4).
- ✅ Aucun sous-lot L3.x partiel : intégralité L3.1 → L3.9 livrée.
- ✅ Suspensions L3.3 (`apply_detections_fast`) et L3.4 (branches `source="fast"`) tranchées via audit L3.8.
- ✅ Arbitrage `ncc_score` (L3.6) tranché via L3.8.
- ✅ Tag git `b04b-L3`.

---

## 🟣 Lot L4 — Validation intégration (smoke test 60 s)

### L4.1 — Smoke test 60 s

- Lancer l'application en mode capture standard pendant **60 s exactement**.
- Vérifier que **chaque** sonde déclarée émet au moins 1 sample (à l'exception de celles conditionnelles à un événement rare — à documenter dans le récap L5).

### L4.2 — Sanity check métriques

- **FPS** : delta < 5% vs baseline pré-L3.
- **Lifetime médian masks** : cohérent avec mesure B-02 (~1.5 s).
- **Bursts CREATE/s** : cohérent avec instrumentation B-05 (déjà active).
- **Aucune exception** dans les logs.

### L4.3 — Validation hot-reload

- Modifier 1 clé config, sauver `config.yaml`.
- Vérifier que `FastTrackThread.maybe_reload()` log le reload sans erreur.
- Vérifier que la sonde `fast_hot_reload` (L3.8) s'incrémente.

### L4.4 — Validation sortie 2 fichiers JSONL (Option C)

- Vérifier que `bench_frame.jsonl` reçoit 1 ligne / frame.
- Vérifier que `bench_agg.jsonl` reçoit 1 ligne / `agg_interval`.
- Vérifier **structure par domaine** : chaque ligne contient au moins les clés racine attendues (`ts`, `frame_id` pour frame ; `ts`, `window_s` pour agg) + ≥ 1 domaine peuplé.
- Vérifier qu'aucun domaine vide n'est sérialisé (`"motion": {}` interdit).
- Vérifier cohérence keys normalisées (T-1 / T-2 / L0.4).

### 🔒 Borne L4

- ✅ Toutes sondes émettent (ou sont documentées comme conditionnelles).
- ✅ Aucune régression mesurable.
- ✅ Hot-reload fonctionnel.
- ✅ 2 fichiers JSONL bien alimentés, schéma L0.4 respecté.
- ✅ Tag git `b04b-L4`.

---

## ⚪ Lot L5 — Livraison B-04b

### L5.1 — Récap consolidé écrit

- Produire le livrable markdown final de B-04b :
  - Liste exhaustive des 85 sondes finales (nom, type, fichier, **domaine JSON**, justification).
  - Liste des décisions transverses (3) appliquées (Option C / `snapshot_*` / migration config).
  - **Schéma JSONL L0.4 figé** (annexe normative).
  - Liste des dettes purgées (4 : `_motion_stats`, `frame_count/fps_timer`, `_b01_stats`, et `tracker.stats()` tracée explicitement).
  - Liste des arbitrages structurels tranchés (1 : `created_ts` Option A).
  - Liste des suspensions levées (2 : `apply_detections_fast` + branches `source="fast"` associator).
  - Liste des sondes écartées avec motifs (référence audit par fichier).
- **Destination** : `Plan_Tracker.md` (section B-04b clôturée).

### L5.2 — Mise à jour `Plan_Tracker.md`

- Marquer B-04b ✅ livré.
- Lever blocage B-04c, B-04d, B-05.

### L5.3 — Bascule B-04c

- Précondition activée : _« Session > 30 s avec variance FPS mesurée séparément scène avec/sans masks actifs. »_
- Démarrer la session de référence.

### 🔒 Borne L5

- ✅ Récap consolidé écrit et archivé.
- ✅ Plan_Tracker.md à jour.
- ✅ B-04c prêt à démarrer.
- ✅ Tag git `b04b-L5`.

---

## 🛑 Règles transverses non-négociables

1. **Aucun lot ne démarre tant que le précédent n'est pas validé** (borne 🔒 cochée).
2. **Aucune sonde ajoutée hors plan** → backlog B-04c.
3. **Rollback** : tag git à chaque borne 🔒. Granularité commits intra-lot libre.
4. **Pas de mise en œuvre avant validation du plan v4**.
5. **Aucune optimisation opportuniste** : tout écart va en P-04.
6. **L3 atomique vis-à-vis de L4** : L3.1 → L3.9 tous livrés avant L4.
7. **L3.8 prioritaire dans L3** : lève 3 suspensions (L3.3, L3.4, L3.6).
8. **Schéma L0.4 immuable** : toute évolution post-L0 = nouveau ticket.
9. **Nommage T-2 = contrat de groupement JSON** : toute nouvelle sonde doit respecter `<domaine>_<action>[_<qualifieur>]` sous peine de tomber dans un domaine inattendu.

---

## 📊 Synthèse finale post-L5

| Métrique                         | Valeur cible                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------ |
| Sondes actives totales           | 85 (60 nouvelles + 26 préservées − 1 supprimée)                                |
| Sondes écartées (motifs tracés)  | ≥ 23                                                                           |
| Décisions transverses appliquées | 3 / 3                                                                          |
| Dettes legacy purgées            | 4 / 4                                                                          |
| Arbitrages structurels tranchés  | 1 / 1 (Option A `created_ts`)                                                  |
| Suspensions levées               | 2 / 2 (via L3.8)                                                               |
| Régression FPS tolérée           | < 5%                                                                           |
| Durée smoke test L4              | 60 s                                                                           |
| Fichiers JSONL en sortie         | 2 (`bench_frame.jsonl` + `bench_agg.jsonl`)                                    |
| Domaines JSON                    | 8 (`main`, `tracker`, `registry`, `motion`, `detect`, `boxes`, `fast`, `mask`) |
| `agg_interval` par défaut        | 1.0 s                                                                          |
| État B-04b                       | ✅ Livré                                                                       |
| État B-04c                       | 🟢 Démarrable                                                                  |

---

## 🚦 En attente

Validation du plan **v4** tel quel avant lancement L0, ou demande d'ajustement final.
