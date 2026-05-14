# 📋 Plan de mise en œuvre séquencé — Post-audit B-04b (v3)

> **Objectif** : passer du backlog audité (60 sondes différées + 3 décisions transverses + 3 dettes legacy + 1 arbitrage structurel) à un système instrumenté, propre et prêt pour B-04c.
>
> **Principe directeur** : séquentialité stricte, chaque lot indépendant testable, rollback possible à chaque borne. Aucun lot ne démarre tant que le précédent n'est pas validé.
>
> **Arbitrages v2 → v3** :
>
> - L2.3 (`tracker.stats()`) : décision différée au moment de L2.
> - L3 : granularité commits déléguée à l'équipe d'implémentation.
> - L4 : smoke test 60 s confirmé.
> - **v3** : intégration exhaustive du détail audit par fichier (sondes, écartées, suspensions, dettes, statu quo) dans L3.

---

## 🎯 Vue d'ensemble

| Lot    | Scope                                                               | Effort  | Précondition           |
| ------ | ------------------------------------------------------------------- | ------- | ---------------------- |
| **L0** | Fondations transverses (Option C + `snapshot_*` + migration config) | ~2 h    | Audit B-04b clôturé ✅ |
| **L1** | Arbitrage structurel : `created_ts` natif sur `Mask`                | ~30 min | L0                     |
| **L2** | Purge dettes legacy (3 blocs)                                       | ~1 h    | L1                     |
| **L3** | Déploiement 60 sondes différées (par fichier)                       | ~3 h    | L2                     |
| **L4** | Validation intégration (smoke test 60 s)                            | ~30 min | L3                     |
| **L5** | Livraison B-04b : récap consolidé + bascule B-04c                   | ~30 min | L4                     |

**Effort total estimé** : ~7 h 30, étalable sur 2 sessions.

---

## 🔵 Lot L0 — Fondations transverses

> ⚠️ **Lot critique** : modifie l'API `bench` et la structure config. Tout ce qui suit dépend de ces fondations. À traiter en premier, en un seul commit atomique.

### L0.1 — API `bench` : `snapshot_all()` + `snapshot_frame()`

> 🔁 **Recadrage v3** : la dette initialement nommée `snapshot_and_reset()` a été **requalifiée** lors de l'audit (décision actée n°2). Deux primitives distinctes :

- **Fichier** : `bench.py`
- **Actions** :
  - `snapshot_all() -> dict` : snapshot **glissant** (cumulatif depuis start) destiné à `bench_agg.jsonl`.
  - `snapshot_frame() -> dict` : snapshot **par frame** destiné à `bench_frame.jsonl`.
- **Contrat** : thread-safe, idempotent sur snapshot vide. Le reset n'est plus implicite — la sémantique cumulative est explicite.
- **Validation** : tests unitaires (snapshot vide, snapshot non-vide, concurrence basique, cohérence cumul/différentiel).

### L0.2 — Option C : décisions transverses actées

- **T-1 (Option C)** : 2 fichiers JSONL en sortie bench :
  - `bench_frame.jsonl` (1 ligne / frame).
  - `bench_agg.jsonl` (1 ligne / période agrégée).
- **T-2 (nommage sondes)** : convention verrouillée `<domaine>_<action>[_<qualifieur>]`.
- **T-3 (rétention)** : cumulatif depuis start, snapshot périodique différentiel pour `bench_agg.jsonl`.
- **Fichiers touchés** : `bench.py`, `main.py` (consommateur des deux snapshots).

### L0.3 — Migration config `debug.csv.*` → `debug.bench.jsonl.*`

> 📦 **Décision actée n°3** : alignement de la nomenclature config avec l'Option C.

- **Fichier** : `config.yaml` + `config.py`.
- **Actions** :
  - Renommage du bloc `debug.csv.*` → `debug.bench.jsonl.*`.
  - Conservation sémantique des sous-clés (`per_frame`, `aggregated`, `mask`, `agg_interval`, `fast`, `fast_interval`).
  - Verrouillage invariants A-02 catégories 3 & 8 dans `config.py` (validation au chargement).
- **Validation** : test de chargement avec config volontairement invalide (doit lever erreur explicite).

### 🔒 Borne L0

- ✅ `bench.snapshot_all()` + `bench.snapshot_frame()` opérationnels + testés.
- ✅ Sortie 2 fichiers JSONL (`bench_frame.jsonl` + `bench_agg.jsonl`) opérationnelle.
- ✅ Bloc config renommé `debug.bench.jsonl.*`, validation durcie sans casser la config courante.
- ✅ Aucune sonde nouvelle ajoutée à ce stade.

---

## 🟢 Lot L1 — Arbitrage structurel `created_ts`

> 🎯 **Option A retenue** (cf. audit `tracker/registry.py`) : champ natif sur `Mask`. Préféré à l'Option B (dict `_lifetime_stats` dans registry) car réutilisable futur + nettoyage purge `_b01_stats` (L2).

### L1.1 — Ajout champ `created_ts` à `Mask`

- **Fichier** : `core/mask.py`
- **Action** :

  ```python
    created_ts: float = 0.0
  ```

  - Dans `__post_init__` : `if self.created_ts == 0.0: self.created_ts = self.last_detected_ts`

- **Action `to_dict()`** : exposer `"created_ts": round(self.created_ts, 4)`.

### L1.2 — Documentation contrat `last_seen_ts` ≡ `last_match_ts`

- **Fichier** : `core/mask.py`
- **Action** : docstring explicite sur `transition()` : _« `last_seen_ts` est rafraîchi exclusivement sur `event="matched"` et joue le rôle de `last_match_ts`. Aucun champ distinct n'est requis. »_
- **Conséquence** : la sonde `mask_last_match_age_s` (L3.2) lira `last_seen_ts`, pas un champ dédié.

### L1.3 — Propagation côté registry

- **Fichier** : `tracker/registry.py`
- **Action** : passer `created_ts=ts` explicitement lors de la création d'un mask (sinon fallback `last_detected_ts` via `__post_init__`).

### 🔒 Borne L1

- ✅ Champ `created_ts` accessible sur tous les masks.
- ✅ `to_dict()` cohérent.
- ✅ Sémantique `last_seen_ts` ≡ `last_match_ts` documentée.
- ✅ Aucune régression sur session courte (smoke test).

---

## 🟡 Lot L2 — Purge dettes legacy

> ⚠️ Ordre impératif : purger **avant** d'ajouter les sondes neuves, sinon double instrumentation transitoire et bruit dans les métriques.

### L2.1 — Dette `_motion_stats` + `get_and_reset_stats()`

- **Fichier** : `tracker/motion.py`
- **Action** : supprimer `_motion_stats` (variable module) et `get_and_reset_stats()`.
- **Consommateur** : `main.py` → remplacer par lecture `bench` directe.

### L2.2 — Dette `frame_count` / `fps_timer` legacy

- **Fichier** : `main.py`
- **Action** : retirer le bloc legacy de comptage FPS hors-bench, remplacer par lecture `bench.snapshot_all()` / `snapshot_frame()`.

### L2.3 — Dette `tracker.stats()` — **DÉCISION DIFFÉRÉE**

- **Statut** : arbitrage à mener **au moment d'attaquer L2**.
- **Procédure** :
  1. Grep `tracker.stats(` dans la codebase (5 min max).
  2. Si seul appelant = `main.py` → supprimer, migrer vers lecture bench directe.
  3. Sinon (consommateurs multiples ou HUD direct) → **conserver en l'état**, scoper en P-04 ultérieur.

### L2.4 — Dette `_b01_stats` (registry) — **bloc post-probes**

> 📦 **Bloc dette acté en étape 3** (audit `tracker/registry.py`). À solder en L2, **avant** L3.2 (les sondes du registry dépendent de la purge).

- **Fichier** : `tracker/registry.py`
- **Actions** :
  - Suppression complète `_b01_stats` (dict) + initialisation `__init__`.
  - Suppression `_b01_tick_counter`.
  - Suppression `_b01_zombie_threshold_s`.
  - Suppression `import time` (plus utilisé après purge).
  - Suppression logs `[B01] CREATE/...`.
  - Suppression bloc "ZOMBIE-SUSPECT" (échantillonnage tick%30, redondant avec probes L3.2).
  - Suppression commentaires `# QUICK-FIX B-01:`.
- **Précondition implicite** : L1 livré (le calcul `mask_lifetime_s` à l'EXPIRE s'appuie désormais sur `created_ts` natif).

### 🔒 Borne L2

- ✅ `_motion_stats` et `get_and_reset_stats()` supprimés.
- ✅ `frame_count`/`fps_timer` legacy retirés de `main.py`.
- ✅ Décision `tracker.stats()` tracée explicitement (supprimé OU scopé P-04).
- ✅ Bloc `_b01_stats` intégralement purgé de `registry.py` (dict, compteurs, logs, imports, commentaires).
- ✅ `main.py` lit exclusivement via `bench`.
- ✅ Smoke test : HUD FPS toujours cohérent.

---

## 🔴 Lot L3 — Déploiement des 60 sondes différées

> 📐 **Stratégie** : déploiement organisé **par fichier** (9 sous-lots logiques L3.1 → L3.9). La **granularité des commits** est laissée à l'équipe d'implémentation.
>
> **Contrainte non-négociable** : l'intégralité de L3 doit être livrée avant la borne L4 (pas de smoke test partiel).
>
> **Légende statuts** :
>
> - 🔒 Différé : à poser pendant L3.
> - 🟡 Suspendu : conditionné à un autre arbitrage (audit `FastTrackThread`).
> - 🟢 Statu quo : conservé, aucune action.
> - ❌ Écarté : non retenu (motif tracé).

---

### L3.1 — `main.py` (5 sondes)

| Sonde             | Type  | Emplacement                                                               |
| ----------------- | ----- | ------------------------------------------------------------------------- |
| `frame_dropped`   | count | §1 — branche `if frame is None` avant `continue`                          |
| `give_frame_slow` | count | §2 — après `detector.give_frame(...)`                                     |
| `give_frame_fast` | count | §2 — après `fast_tracker.give_frame_and_views(...)` (si snapshot)         |
| `snapshot_build`  | timer | §2 — englobant `snapshot = [...]` + `views = [...]`                       |
| `loop_total`      | timer | Englobant l'itération complète `while True:` (de `maybe_reload` à fin §8) |

**Existant** : 9 sondes conformes au plan, conservées.

---

### L3.2 — `tracker/registry.py` (7 sondes)

> **Précondition dure** : L2.4 (purge `_b01_stats`) livré. L1 (`created_ts`) livré.

| Sonde                   | Type  | Emplacement                                                                                    |
| ----------------------- | ----- | ---------------------------------------------------------------------------------------------- |
| `mask_created`          | count | `create()` — après `_add(mask)`                                                                |
| `mask_expired`          | count | `tick_and_expire()` — branche purge LOST, avant `del self._masks[uid]`                         |
| `mask_evicted`          | count | `_evict_one()` — avant `del self._masks[worst.uid]`                                            |
| `mask_lifetime_s`       | probe | `tick_and_expire()` — au moment de l'EXPIRE (valeur = `ts - created_ts`)                       |
| `mask_transition_lost`  | count | `tick_and_expire()` — après `mask.transition("missing", ts)` si transition effective vers LOST |
| `mask_last_match_age_s` | probe | `tick_and_expire()` — au moment de l'EXPIRE (valeur = `ts - last_seen_ts`)                     |
| `mark_matched`          | count | `mark_matched()` — labellisé par source (`slow` / `fast` / `unknown`)                          |

**Existant** : 0 sonde. Gauges `masks_*` déléguées à `Tracker.tick()` (cf. L3.3).

---

### L3.3 — `tracker/tracker.py` (8 sondes)

| Sonde                   | Type  | Emplacement                                                        | Label / valeur                       |
| ----------------------- | ----- | ------------------------------------------------------------------ | ------------------------------------ |
| `apply_detections_slow` | timer | `apply_detections()` englobant, **uniquement si `source=="slow"`** | —                                    |
| `apply_fast_direct`     | timer | `apply_fast_direct()` englobant                                    | —                                    |
| `fast_received`         | count | `apply_fast_direct()` — `+= len(uid_to_rect)`                      | —                                    |
| `fast_applied`          | count | `apply_fast_direct()` — `+= len(matched_uids)`                     | —                                    |
| `fast_drift_skipped`    | count | `apply_fast_direct()` — `+= drift_skipped`                         | —                                    |
| `fast_drift_max_s`      | probe | `apply_fast_direct()` — `drift_max_seen` si `drift_skipped > 0`    | —                                    |
| `detection_matched`     | count | `apply_detections()` — `+= len(matches)`                           | labellisé `source` (`slow` / `fast`) |
| `detection_unmatched`   | count | `apply_detections()` — `+= len(unmatched_dets)`                    | labellisé `source` (`slow` / `fast`) |

**Suspendu 🟡** :

| Sonde                   | Condition de déblocage                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `apply_detections_fast` | Audit `FastTrackThread` (L3.8) : confirmer si la branche `source=="fast"` dans `apply_detections()` est encore appelée. |
|                         | Si code mort → suppression branche + sonde écartée.                                                                     |

**Dettes** :

- ❌ Log DEBUG `[FAST-APPLY] received=... applied=... drift_skipped=... max_drift=...` → suppression (redondant avec probes).
- 🟢 Log WARNING `[FAST-APPLY] drift dégradé` → conservé (alerte humaine).
- 🟢 Variables `drift_skipped`, `drift_max_seen` → conservées (alimentent counts + probe + WARNING).

**Écartées ❌** : `tick_duration` (doublon `bench.timer("predict")`), `predict_position_calls` (low value), `cfg_reload` (déjà loggé INFO).

**Existant** : 3 gauges conformes (`masks_confirmed`, `masks_pending`, `masks_lost`).

---

### L3.4 — `tracker/associator.py` (8 sondes)

| Sonde                     | Type  | Emplacement                                                      | Label / valeur                       |
| ------------------------- | ----- | ---------------------------------------------------------------- | ------------------------------------ |
| `cost_matrix_size`        | probe | `_build_cost_matrix()` — entrée, `n_det * n_mask`                | —                                    |
| `geo_gate_passed`         | count | `_geo_gate_passes()` — branche `passes==True`                    | —                                    |
| `geo_gate_rejected`       | count | `_geo_gate_passes()` — branche `passes==False`                   | —                                    |
| `min_score_rejected`      | count | `_build_cost_matrix()` — branche `ms.total < min_score`          | —                                    |
| `match_score`             | probe | `associate()` — commit `matches`, valeur `ms.total`              | labellisé `source` (`slow` / `fast`) |
| `match_iou_component`     | probe | `associate()` — commit `matches`, valeur `ms.iou`                | —                                    |
| `match_hash_component`    | probe | `associate()` — commit `matches`, valeur `ms.hash` (si non-None) | —                                    |
| `match_continuity_factor` | probe | `associate()` — commit `matches`, valeur `ms.continuity`         | —                                    |

**Suspendu 🟡** :

| Élément                                                  | Condition de déblocage                                                              |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Branche`source == "fast"` dans                           | Audit `FastTrackThread` (L3.8) + clôture suspension `apply_detections_fast` (L3.3). |
| `_get_weights`/`_get_min_score`/`_get_source_confidence` | Si jamais appelé avec `source="fast"` → suppression branches + arbitrage config     |
|                                                          | (`weights_source_fast` / `match_score_min_fast` / `source_confidence_fast`).        |

**Statu quo 🟢** : Logger `"associator"` déclaré inutilisé — conservé.

**Écartées ❌** : `associate_duration` (doublon `apply_detections_slow`), `associate_calls` (déductible), `match_gated_post_hungarian` (low value), `match_score_below_min_post_hungarian` (équivalent `min_score_rejected`).

**Existant** : 0 sonde.

---

### L3.5 — `tracker/motion.py` (5 sondes)

> **Précondition dure** : L2.1 (purge `_motion_stats`) livré.

| Sonde                      | Type  | Emplacement                                                     |
| -------------------------- | ----- | --------------------------------------------------------------- |
| `dead_zone_hit`            | count | `apply_detection()` — branche dead zone position (1ère branche) |
| `velocity_reset_dt_slow`   | count | `apply_detection()` — branche `dt > config.dt_slow_max`         |
| `velocity_reset_teleport`  | count | `apply_detection()` — branche `dist > config.teleport_thresh`   |
| `velocity_magnitude`       | probe | `apply_detection()` post-clamp source=="slow", `sqrt(vx²+vy²)`  |
| `predicted_clamped_screen` | count | `predict_position()` — si `x` ou `y` clampé aux bornes écran    |

**Statu quo 🟢** : Logger `"motion"` inutilisé, commentaire `# Sondes bench (nouvelles)` — conservés.

**Écartées ❌** : `apply_detection_duration` (doublon), `velocity_clamped_x/y/w/h` (non prioritaire), `predict_position_calls` (déductible), `damping_factor` (violation site unique), `dt_predict_ms` (doublon `staleness_slow_ms`), `predict_cold_start` (low value).

**Existant** : 2 sondes B-04b conformes (`staleness_slow_ms`, `staleness_capped`).

---

### L3.6 — `detection/detect.py` (3 sondes)

| Sonde                      | Type  | Emplacement                                              |
| -------------------------- | ----- | -------------------------------------------------------- |
| `detect_pipeline_duration` | timer | `_run_pipeline()` englobant intégral                     |
| `detect_candidates_count`  | probe | `_run_pipeline()` après `process_channel`, avant remap   |
| `detect_zero_candidates`   | count | `_run_pipeline()` branche `if not candidates: return []` |

**En attente arbitrage transverse** :

| Sonde       | Motif                                                                                      |
| ----------- | ------------------------------------------------------------------------------------------ |
| `ncc_score` | Décision reportée à L3.8 (`fast_track_thread.py`) — risque doublon à arbitrer en contexte. |

**Statu quo 🟢** : Logger `"detect"` inutilisé — conservé.

**Écartées ❌** : `detect_kernel_cache_miss` (low value), `ncc_match_success` (doublon attendu), `ncc_match_below_threshold` (doublon), `detect_plates_clamped` (low value), `detect_template_none` (redondant).

**Existant** : 1 sonde conforme (`fast_ncc_roi_too_small`).

---

### L3.7 — `detection/boxes.py` — `process_channel` (10 sondes)

> Périmètre : orchestrateur `process_channel` uniquement. Sous-modules `mask.py` / `tools.py` hors scope L3.

| Sonde                         | Type  | Emplacement                                           |
| ----------------------------- | ----- | ----------------------------------------------------- |
| `boxes_pipeline`              | timer | Englobant tout `process_channel`                      |
| `boxes_validate_bg`           | timer | Autour de `_validate_background`                      |
| `boxes_validate_text`         | timer | Autour de `_validate_text`                            |
| `boxes_n_raw`                 | probe | Après `_extract_raw_boxes` — `len(boxes)`             |
| `boxes_n_after_geometry`      | probe | Après `_filter_geometry` — `len(geometryed)`          |
| `boxes_n_after_validate_bg`   | probe | Après `_validate_background` — `len(validated_b)`     |
| `boxes_n_after_validate_text` | probe | Après `_validate_text` — `len(validated_t)`           |
| `boxes_n_after_bands`         | probe | Après `_filter_horizontal_bands` — `len(banded)`      |
| `boxes_n_after_alignment`     | probe | Après `_filter_horizontal_alignment` — `len(aligned)` |
| `boxes_n_final`               | probe | Après `_filter_perspective_gradient` — `len(plates)`  |

**Écartées ❌** : `boxes_extract_raw` / `boxes_adjust_resolve_1` / `boxes_split_wide` / `boxes_filter_bands` / `boxes_filter_alignment` / `boxes_filter_perspective` (couverts par `boxes_pipeline` + probes), `boxes_n_after_split` / `boxes_n_after_merge` (peu discriminants).

**Existant** : 0 sonde.

---

### L3.8 — `threads/fast_track_thread.py` (8 sondes)

> 🔑 **Sous-lot pivot** : son audit lève les suspensions L3.3 (`apply_detections_fast`) et L3.4 (branches `source="fast"` dans associator), et tranche le doublon `ncc_score` L3.6.

- 8 sondes nouvelles + suppression sonde `fast_margin` (validée pour suppression en audit étape 9).
- Inclut `fast_ncc_score` (résolution arbitrage transverse `ncc_score`).
- Inclut `fast_hot_reload` (count) — alimente vérification L4.3.

> Détail exhaustif des 8 sondes à reprendre depuis le récap d'audit étape 9 (`fast_track_thread.py`).

---

### L3.9 — `core/mask.py` (6 sondes)

| Sonde                                | Type  | Emplacement                                                     |
| ------------------------------------ | ----- | --------------------------------------------------------------- |
| `mask_transition_matched`            | count | `transition()` — branche `event=="matched"`                     |
| `mask_transition_missing`            | count | `transition()` — branche `event=="missing"`                     |
| `mask_promoted_pending_to_confirmed` | count | `transition()` — branche `PENDING → CONFIRMED` (matched)        |
| `mask_resurrected_lost_to_confirmed` | count | `transition()` — branche `LOST → CONFIRMED` (matched)           |
| `mask_transition_to_lost`            | count | `transition()` — branche `→ LOST` (missing avec quorum atteint) |
| `mask_to_dict_calls`                 | count | `to_dict()` — entrée méthode (surveillance overhead export)     |

**Existant** : 0 sonde (dataclass pur, conforme).

---

### 🔒 Borne L3

- ✅ 60 sondes nouvelles actives (5 + 7 + 8 + 8 + 5 + 3 + 10 + 8 + 6 = **60**).
- ✅ 26 sondes existantes préservées.
- ✅ 1 sonde supprimée (`fast_margin`).
- ✅ Bench JSONL contient les 85 sondes attendues sur session courte.
- ✅ Aucune régression FPS (delta < 5% — validé en L4).
- ✅ Aucun sous-lot L3.x partiel : intégralité L3.1 → L3.9 livrée.
- ✅ Suspensions L3.3 (`apply_detections_fast`) et L3.4 (branches `source="fast"`) tranchées via audit L3.8.
- ✅ Arbitrage `ncc_score` (L3.6) tranché via L3.8.

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

### L4.4 — Validation sortie 2 fichiers JSONL

- Vérifier que `bench_frame.jsonl` reçoit 1 ligne / frame.
- Vérifier que `bench_agg.jsonl` reçoit 1 ligne / `agg_interval`.
- Vérifier cohérence keys normalisées (T-1 / T-2).

### 🔒 Borne L4

- ✅ Toutes sondes émettent (ou sont documentées comme conditionnelles).
- ✅ Aucune régression mesurable.
- ✅ Hot-reload fonctionnel.
- ✅ 2 fichiers JSONL bien alimentés selon Option C.

---

## ⚪ Lot L5 — Livraison B-04b

### L5.1 — Récap consolidé écrit

- Produire le livrable markdown final de B-04b :
  - Liste exhaustive des 85 sondes finales (nom, type, fichier, justification).
  - Liste des décisions transverses (3) appliquées (Option C / `snapshot_*` / migration config).
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

---

## 🛑 Règles transverses non-négociables

1. **Aucun lot ne démarre tant que le précédent n'est pas validé** (sa borne 🔒 doit être cochée).
2. **Aucune sonde ajoutée hors plan** (toute sonde supplémentaire identifiée en cours de route → backlog B-04c).
3. **Rollback** : tag git à chaque borne 🔒 (L0, L1, L2, L3, L4, L5). Granularité commits intra-lot libre.
4. **Pas de mise en œuvre avant validation du plan** : tu valides ce plan v3, puis seulement on démarre L0.
5. **Aucune optimisation opportuniste** : si pendant L3.5 une optim motion est identifiée, elle va en P-04, pas dans L3.5.
6. **L3 atomique vis-à-vis de L4** : quelle que soit la granularité commits choisie, L4 ne démarre que lorsque L3.1 → L3.9 sont **tous** livrés.
7. **L3.8 prioritaire dans L3** : son audit lève 3 suspensions (L3.3, L3.4, L3.6). À traiter **en premier** dans L3 si l'équipe découpe en plusieurs commits.

---

## 📊 Synthèse finale post-L5

| Métrique                         | Valeur cible                                    |
| -------------------------------- | ----------------------------------------------- |
| Sondes actives totales           | 85 (60 nouvelles + 26 préservées − 1 supprimée) |
| Sondes écartées (motifs tracés)  | ≥ 23 (cf. récap audit)                          |
| Décisions transverses appliquées | 3 / 3                                           |
| Dettes legacy purgées            | 4 / 4 (dont L2.3 tracée explicitement)          |
| Arbitrages structurels tranchés  | 1 / 1 (Option A `created_ts`)                   |
| Suspensions levées               | 2 / 2 (via L3.8)                                |
| Régression FPS tolérée           | < 5%                                            |
| Durée smoke test L4              | 60 s                                            |
| Fichiers JSONL en sortie         | 2 (`bench_frame.jsonl` + `bench_agg.jsonl`)     |
| État B-04b                       | ✅ Livré                                        |
| État B-04c                       | 🟢 Démarrable                                   |

---
