# Sondes bench par fichier

[← Retour au README](./../README.md)

Référentiel exhaustif des sondes émises par chaque module instrumenté de l'application RL-NameBlur. Chaque section couvre un fichier ou un domaine fonctionnel.

## Sommaire

- [`main.py` — boucle principale](#mainpy--boucle-principale)
- [Domaine `registry` — `tracker/registry.py`](#domaine-registry--trackerregistrypy)
- [Domaine `tracker` — `tracker/tracker.py`](#domaine-tracker--trackertrackerpy)
- [Domaine `associator` — `tracker/associator.py`](#domaine-associator--trackerassociatorpy)
- [Domaine `motion` — `tracker/motion.py`](#domaine-motion--trackermotionpy)
- [Domaine `capture` — `threads/capture_thread.py`](#domaine-capture--threadscapture_threadpy)
- [Domaine `fast` — `threads/fast_track_thread.py`](#domaine-fast--threadsfast_track_threadpy)
- [Domaine `mask` — `core/mask.py`](#domaine-mask--coremaskpy)
- [Domaine `detect` — `detection/detect.py`](#domaine-detect--detectiondetectpy)

---

## `main.py` — boucle principale

| Sonde                  | Type  | Description                                                | Conditionnel                                 |
| ---------------------- | ----- | ---------------------------------------------------------- | -------------------------------------------- |
| `main_capture_wait_ms` | probe | Durée attente frame capturée (`capturer.get_frame()`)      | Non                                          |
| `main_loop_ms`         | probe | Durée totale traitement frame (étapes 2 → 7)               | Non                                          |
| `main_distribute_ms`   | probe | Durée distribution frame aux threads (détecteur + fast)    | Non                                          |
| `main_frame_id`        | gauge | Identifiant de la frame courante                           | Non                                          |
| `main_slow_poll_ms`    | probe | Durée poll résultat `DetectThread`                         | Non                                          |
| `main_copy_ms`         | probe | Durée copie frame source → buffer SendThread (`np.copyto`) | Non                                          |
| `main_match_ms`        | probe | Durée matching détections slow → tracker                   | Oui — si `slow_updated and new_plates`       |
| `main_fast_poll_ms`    | probe | Durée poll résultat `FastTrackThread`                      | Oui — si `fast_enabled and not slow_updated` |
| `main_predict_ms`      | probe | Durée `tracker.tick()` (predict + TTL + purge)             | Non                                          |
| `main_prepare_ms`      | probe | Durée préparation blur_zones + emprunt buffer SendThread   | Non                                          |
| `main_blur_ms`         | probe | Durée floutage + overlay debug                             | Non                                          |
| `main_send_ms`         | probe | Durée publication frame vers `SendThread`                  | Non                                          |
| `main_stats_ms`        | probe | Durée mise à jour compteurs/gauges de fin de boucle        | Non                                          |
| `main_frames_total`    | count | Nombre cumulé de frames traitées                           | Non                                          |
| `main_masks_total`     | gauge | Nombre de masques confirmés à la frame courante            | Non                                          |

> `main_loop_ms` est la référence du frame budget (`FRAME_BUDGET_REFERENCE` dans `bench/compare/_config.py`). Son périmètre couvre les étapes 2 à 7 de la boucle (distribution threads → publication frame). `main_capture_wait_ms` est mesuré **hors** `main_loop_ms` (étape 1, attente I/O) et n'entre donc pas dans le budget.
> `main_blur_ms` couvre floutage et overlay debug dans le même `bench.timer`.`tracker_confirmed`, `tracker_pending`, `tracker_lost` sont lus dans `main.py` via `bench.read_gauge()` pour le log console uniquement — ces gauges sont posées dans `tracker.tick()` et documentées dans le domaine `tracker`.`motion_staleness_slow_ms` est lu via `bench.last()` pour le log console uniquement posé dans `motion.py` et documenté dans le domaine `motion`. `bench.push_frame()` appelé en fin de boucle — infrastructure bench, pas une sonde métier.

## Domaine `registry` — `tracker/registry.py`

Sondes émises par `MaskRegistry`. Les gauges sont recalculées
à chaque appel `tick_and_expire()`, en fin de boucle après
toutes les transitions.

| Nom                     | Type  | Description                                                | Conditionnel                                      |
| ----------------------- | ----- | ---------------------------------------------------------- | ------------------------------------------------- |
| `registry_create_total` | count | Incrémenté à chaque création de `Mask`                     | Non                                               |
| `registry_lost_total`   | count | Incrémenté à chaque transition → LOST                      | Oui — si mask non matché depuis > `lost_after_s`  |
| `registry_expire_total` | count | Incrémenté à chaque purge de mask LOST                     | Oui — si mask LOST depuis > `expire_after_lost_s` |
| `registry_evict_total`  | count | Incrémenté à chaque éviction capacité max                  | Oui — si `len(masks) >= max_masks`                |
| `registry_confirmed`    | gauge | Nombre de masks en état CONFIRMED (instantané fin de tick) | Non                                               |
| `registry_pending`      | gauge | Nombre de masks en état PENDING (instantané fin de tick)   | Non                                               |
| `registry_lost`         | gauge | Nombre de masks en état LOST (instantané fin de tick)      | Non                                               |

> Les 4 counts (`_total`) sont cumulatifs depuis le démarrage de la session.
> `registry_lost` (gauge) et `registry_lost_total` (count) sont orthogonaux :
> la gauge reflète le stock instantané après purge, le count l'historique cumulé des transitions avant éventuelle purge.
> `_evict_one()` est appelée avant insertion — `registry_evict_total` est émis avant que le nouveau mask entre dans le dict.
> Les gauges d'état sont posées après toutes les transitions du tick — instantané cohérent de fin de cycle.
> Ces gauges (`registry_confirmed/pending/lost`) sont distinctes des gauges
> `tracker_confirmed/pending/lost` posées dans `tracker.tick()`.
> Les `registry_*` sont destinées au JSONL et aux analyses post-session.
> Les `tracker_*` sont la source de vérité consommée par `main.py` au runtime.

## Domaine `tracker` — `tracker/tracker.py`

| Sonde                              | Type  | Description                                                       | Conditionnel                 |
| ---------------------------------- | ----- | ----------------------------------------------------------------- | ---------------------------- |
| `tracker_apply_slow_detections_ms` | probe | Durée totale de `apply_slow_detections()` (matching slow → masks) | Non                          |
| `tracker_detections_in`            | count | Nombre de détections reçues par `apply_slow_detections()`         | Non — 0 si aucune détection  |
| `tracker_apply_fast_detections_ms` | probe | Durée totale de `apply_fast_detections()` (mise à jour fast)      | Non                          |
| `tracker_fast_drift_skipped`       | count | Masks ignorés pour drift excessif lors du fast update             | Oui — si `drift_skipped > 0` |
| `tracker_tick_ms`                  | probe | Durée totale de `tick()` (predict + TTL + purge)                  | Non                          |
| `tracker_confirmed`                | gauge | Nombre de masks CONFIRMED (instantané fin de tick)                | Non                          |
| `tracker_pending`                  | gauge | Nombre de masks PENDING (instantané fin de tick)                  | Non                          |
| `tracker_lost`                     | gauge | Nombre de masks LOST (instantané fin de tick)                     | Non                          |
| `tracker_masks_total`              | gauge | Nombre total de masks actifs (tous états)                         | Non                          |

> `tracker_confirmed`, `tracker_pending`, `tracker_lost` sont lus par `main.py`
> via `bench.read_gauge()` pour le log console — source de vérité runtime.
> Ces gauges sont distinctes des `registry_confirmed/pending/lost` (voir domaine `registry`).
> `tracker_fast_drift_skipped` est absent du JSONL sur les frames
> où aucun drift n'est détecté.

## Domaine `associator` — `tracker/associator.py`

Sondes émises par le pipeline d'association détections → masks (gating + coût IoU+pHash + Hungarian).

| Sonde                                 | Type  | Description                                                        | Conditionnel                                |
| ------------------------------------- | ----- | ------------------------------------------------------------------ | ------------------------------------------- |
| `associator_tick_ms`                  | probe | Durée totale du pipeline d'association                             | Non                                         |
| `associator_gated_total`              | count | Paires détection×mask écartées par le gate de distance             | Oui — si au moins une paire gated           |
| `associator_candidates_total`         | count | Paires détection×mask évaluées après gating                        | Non — 0 si matrice vide                     |
| `associator_matched_total`            | count | Paires retenues par l'algorithme hongrois                          | Non — 0 si aucun match                      |
| `associator_hungarian_rejected_total` | count | Paires rejetées par l'hongrois (paire gated + score < `min_score`) | Oui — si au moins un rejet                  |
| `associator_unmatched_det_total`      | count | Détections sans mask associé (nouvelles plaques potentielles)      | Oui — si au moins une détection non matchée |
| `associator_unmatched_mask_total`     | count | Masks sans détection associée (candidats LOST)                     | Oui — si au moins un mask non matché        |
| `associator_score_rejected_total`     | count | Paires rejetées pour score total insuffisant (`< min_score`)       | Oui — si au moins un rejet score            |
| `associator_reject_in_lost_window`    | count | Rejets score pour masks en état LOST (post-gating)                 | Oui — si au moins un tel rejet              |

> `associator_gated_total` et `associator_candidates_total` sont complémentaires : `gated + candidates = N × M` paires totales évaluées.
> `associator_hungarian_rejected_total` agrège deux causes distinctes (paire gated et score total < `min_score`) en un seul compteur — non désagrégeable sans modification du code d'émission.
> `associator_score_rejected_total` permet d'isoler les rejets score seuls, orthogonal à `associator_hungarian_rejected_total`.
> Les counts conditionnels sont absents du JSONL sur les frames où leur branche n'est pas atteinte.
> `associator_reject_in_lost_window` est incrémenté dans deux branches : `_build_cost_matrix` (score insuffisant) et `associate()` (hongrois reject).
> Les counts conditionnels sont absents du JSONL sur les frames où leur branche n'est pas atteinte.

## Domaine `motion` — `tracker/motion.py`

Fonctions pures (sans état global). Sondes émises par `apply_detection()` et `predict_position()`.
`compute_predicted_rect()` est intentionnellement sans sonde (appelée N×M fois par l'associator).

| Sonde                        | Type  | Description                                                      | Conditionnel                        |
| ---------------------------- | ----- | ---------------------------------------------------------------- | ----------------------------------- |
| `motion_apply_ms`            | probe | Durée de `apply_detection()` (mise à jour état cinématique)      | Non                                 |
| `motion_predict_ms`          | probe | Durée de `predict_position()` (extrapolation position)           | Non                                 |
| `motion_dt_slow_ms`          | probe | Intervalle de temps réel entre deux détections slow (ms)         | Non                                 |
| `motion_staleness_slow_ms`   | probe | Staleness accumulée depuis la dernière détection slow (ms)       | Oui — si staleness dépasse le seuil |
| `motion_velocity_pps`        | probe | Vitesse courante du mask (pixels/seconde)                        | Non                                 |
| `motion_residual_px`         | probe | Résidu entre position prédite et position détectée (px)          | Non                                 |
| `motion_teleport_total`      | count | Masks dont le saut de position dépasse le seuil de téléportation | Oui — si saut détecté               |
| `motion_dt_clamped_total`    | count | Masks dont `dt` a été capé par `dt_cap`                          | Oui — si dépassement cap            |
| `motion_staleness_capped`    | count | Masks dont `abs(staleness)` > `dt_cap`                           | Oui — si dépassement cap            |
| `motion_predict_source_fast` | count | Prédictions utilisant le triplet FAST (fast_kin)                 | Non                                 |
| `motion_predict_source_slow` | count | Prédictions utilisant le triplet SLOW (last_detected_rect)       | Non                                 |

> `motion_staleness_slow_ms` est alimentée **uniquement** depuis `predict_position()` (1×/mask non matché/tick), pas depuis `compute_predicted_rect()`. Sémantique : fraîcheur de la dernière détection slow, pas fraîcheur du suivi global.
> `motion_velocity_pps` exprime la vitesse en pixels par seconde (norme du vecteur vx/vy).
> `motion_predict_source_*` reflète le même Branchement que `compute_predicted_rect()` pour la sélection du triplet source.
> Les counts conditionnels sont absents du JSONL sur les frames où leur branche n'est pas atteinte.

## Domaine `capture` — `threads/capture_thread.py`

| Sonde              | Type  | Description                                        | Conditionnel                                 |
| ------------------ | ----- | -------------------------------------------------- | -------------------------------------------- |
| `capture_frame_ms` | probe | Durée de `source.grab()` (acquisition frame brute) | Non                                          |
| `capture_drop`     | count | Incrémenté si `source.grab()` retourne `None`      | Oui — uniquement si `source.grab()` → `None` |

> `capture_drop` est absent du JSONL sur les frames où `grab()` retourne une frame valide.

---

## Domaine `fast` — `threads/fast_track_thread.py`

Sondes émises par le thread de suivi inter-frames (Optical Flow + NCC). Chaque sonde `fast_*` a une variante `F_*` **identique en valeur** émise simultanément :
les variantes `F_*` sont **non filtrées** par `_is_fast_probe` et apparaissent aussi dans le canal `frame` (mesures exactes non agrégées). Ce comportement
est intentionnel : les `F_*` fournissent des métriques par frame individualisées.

> **Note — DetectThread non instrumenté** : `DetectThread` (slow) n'émet aucune sonde.Le pipeline slow detect est instrumenté dans `detect.py` (domaine `detect`).

### Sondes `fast_*` (agrégées — canal `fast`)

| Sonde                       | Type  | Description                                           | Conditionnel                |
| --------------------------- | ----- | ----------------------------------------------------- | --------------------------- |
| `fast_wakeup_lag_ms`        | probe | Délai entre signal Event et début de traitement (ms)  | Non                         |
| `fast_tick_total`           | count | Nombre de cycles fast traités                         | Non — 0 si aucune view      |
| `fast_n_masks`              | probe | Nombre de FastMaskView actives à l'entrée du cycle    | Non                         |
| `fast_tick_ms`              | timer | Durée totale du cycle fast track (ms)                 | Non                         |
| `fast_cvt_ms`               | timer | Durée conversion RGB→GRAY (ms)                        | Non                         |
| `fast_of_total_ms`          | timer | Durée du tracking Optical Flow global (ms)            | Non — 0 si aucune view      |
| `fast_mask_processed_total` | count | Vues traitées par le pipeline (OF tenté si template)  | Non — 0 si aucune view      |
| `fast_of_failed_total`      | count | Vues où OF n'a pas réussi à proposer une position     | Oui — OF failed             |
| `fast_ncc_total_ms`         | timer | Durée des appariements NCC (toutes views) (ms)        | Oui — au moins un NCC tenté |
| `fast_margin_ms`            | timer | Durée de calcul de la marge adaptive (ms)             | Oui — au moins un NCC tenté |
| `fast_margin_px`            | probe | Marge adaptive en pixels par view                     | Oui — au moins un NCC tenté |
| `fast_ncc_score`            | probe | Score NCC par view                                    | Oui — au moins un NCC tenté |
| `fast_ncc_confirmed_total`  | count | Vues confirmées par NCC (score ≥ `ncc_v_gate`)        | Oui — NCC réussi            |
| `fast_v_px_per_s`           | probe | Vitesse inter-tick NCC (pixels/seconde, dx/dt, dy/dt) | Oui — NCC confirmé          |
| `fast_mask_lost_total`      | count | Vues perdues (`stale > max_stale_frames`)             | Oui — stale dépassé         |
| `fast_stale_skipped_total`  | count | Vues stale tolérées (`stale ≤ max_stale_frames`)      | Oui — stale toléré          |

### Variantes `F_*` (exactes — canal `frame` uniquement)

| Sonde                    | Type  | Description                             | Canal |
| ------------------------ | ----- | --------------------------------------- | ----- |
| `F_wakeup_lag_ms`        | probe | Identique à `fast_wakeup_lag_ms`        | frame |
| `F_tick_total`           | count | Identique à `fast_tick_total`           | frame |
| `F_n_masks`              | probe | Identique à `fast_n_masks`              | frame |
| `F_mask_processed_total` | count | Identique à `fast_mask_processed_total` | frame |
| `F_of_failed_total`      | count | Identique à `fast_of_failed_total`      | frame |
| `F_ncc_confirmed_total`  | count | Identique à `fast_ncc_confirmed_total`  | frame |
| `F_margin_px`            | probe | Identique à `fast_margin_px`            | frame |
| `F_ncc_score`            | probe | Identique à `fast_ncc_score`            | frame |
| `F_v_px_per_s`           | probe | Identique à `fast_v_px_per_s`           | frame |
| `F_mask_lost_total`      | count | Identique à `fast_mask_lost_total`      | frame |
| `F_stale_skipped_total`  | count | Identique à `fast_stale_skipped_total`  | frame |

> Les sondes `fast_margin_px` / `F_margin_px`, `fast_ncc_score` / `F_ncc_score` sont émises par view — plusieurs émissions possibles par cycle si plusieurs views actives.
> `fast_ncc_confirmed_total` / `F_ncc_confirmed_total` sont des counts cumulatifs session.

### Émissions lifecycle (domaine fast)

Le thread fast émet des événements lifecycle via `bench.emit_lifecycle()` :

| Événement   | Trigger                                 | Source   |
| ----------- | --------------------------------------- | -------- |
| `CONFIRMED` | NCC confirme une FastMaskView           | `"fast"` |
| `LOST`      | FastMaskView dépasse `max_stale_frames` | `"fast"` |

## Domaine `mask` — `core/mask.py`

Sondes émises par `Mask.transition()` uniquement.
`to_dict()`, `to_fast_view()`, `__post_init__()` n'émettent aucune sonde.

### Modèle d'horlogerie

L'application utilise deux bases de temps distinctes. Toute sonde de latence doit utiliser la base **capture** (timestamps `*_frame_ts`), pas `perf_counter`.
Les champs `*_ts` (perf_counter) servent uniquement au **TTL** (perte de vue, expiration).

| Sonde                           | Type  | Description                                                   | Base de temps            | Conditionnel                |
| ------------------------------- | ----- | ------------------------------------------------------------- | ------------------------ | --------------------------- |
| `mask_transition_matched_total` | count | Transitions déclenchées par un match (détection associée)     | (count)                  | Oui — si transition matched |
| `mask_promote_total`            | count | Transitions PENDING → CONFIRMED                               | (count)                  | Oui — si promotion          |
| `mask_confirm_latency_ms`       | probe | Délai entre création PENDING et promotion CONFIRMED (ms)      | **capture** (`frame_ts`) | Oui — si promotion          |
| `mask_revive_total`             | count | Transitions LOST → CONFIRMED (revive)                         | (count)                  | Oui — si revive détecté     |
| `mask_revive_latency_ms`        | probe | Délai entre entrée LOST et revive (ms)                        | **capture** (`frame_ts`) | Oui — si revive détecté     |
| `mask_transition_missing_total` | count | Transitions déclenchées sans match (mask non matché ce cycle) | (count)                  | Oui — si transition missing |
| `mask_to_lost_total`            | count | Transitions vers état LOST                                    | (count)                  | Oui — si transition → LOST  |
| `mask_lost_latency_ms`          | probe | Délai entre dernière détection et transition LOST (ms)        | **capture** (`frame_ts`) | Oui — si transition → LOST  |

> les trois sondes `*_latency_ms` utilisent la base de temps **capture** (`last_seen_frame_ts`, `lost_since_frame_ts`).
> Ne pas confondre avec les timestamps perf_counter (`last_seen_ts`, `lost_since_ts`) utilisés pour le TTL (`lost_after_s`, `expire_after_lost_s`).
> `mask_confirm_latency_ms` et `mask_revive_latency_ms` mesurent des délais calculés à l'instant de la transition — non rétroactifs.
> `mask_transition_matched_total` et `mask_transition_missing_total` sont complémentaires : chaque appel à `transition()` incrémente l'un ou l'autre.
> Les sondes conditionnelles sont absentes du JSONL sur les frames sans transition correspondante.
> `mask_revive_latency_ms` est calculée en base CAPTURE :(detected_frame_ts − prev_lost_since_frame_ts).
> Émise seulement si prev_lost_since_frame_ts is not None. Au match, lost_since_frame_ts est remis à None (symétrie avec lost_since_ts). NE PAS utiliser lost_since_ts (base perf_counter) — cf. Plan_Timer Stratégie 3-A

### Champs Mask liés (Plan_Timer)

| Champ                 | Base de temps | Usage                                                  |
| --------------------- | ------------- | ------------------------------------------------------ |
| `last_seen_ts`        | perf_counter  | TTL : `lost_after_s`, fraîcheur masque                 |
| `lost_since_ts`       | perf_counter  | TTL : `expire_after_lost_s`                            |
| `created_ts`          | perf_counter  | TTL confirm + référence latences capture               |
| `last_seen_frame_ts`  | capture       | Latences revive/confirm (via `last_detected_frame_ts`) |
| `lost_since_frame_ts` | capture       | Latence revive (`prev_lost_since_frame_ts`)            |

### Émissions lifecycle (domaine mask)

| Événement   | Trigger                              | Champs `revived` / `frame_id` |
| ----------- | ------------------------------------ | ----------------------------- |
| `CREATED`   | `registry.create()`                  | `revived = null`              |
| `DETECTED`  | Match slow ou fast (dans tracker.py) | `revived = null`              |
| `CONFIRMED` | `transition(matched)` PENDING→CONF   | `revived = null`              |
| `LOST`      | `transition(missing)` (état normal)  | `revived = null`              |
| `REVIVE`    | `transition(matched)` LOST→CONF      | `revived = True`              |
| `EXPIRED`   | Purge TTL (`tick_and_expire()`)      | `revived = null`              |
| `EVICTED`   | Éjection capacité max (`_evict_one`) | `revived = null`              |

> `revived = True` est calculé dans `emit_lifecycle()` : positionné uniquement si `event == REVIVE` **et** `state_val` avant revive est `PENDING` ou `CONFIRMED`.
> `frame_id` est lu via `getattr(mask, "frame_id", -1)` — valorisé par `tracker.apply_*()`.

## Domaine `detect` — `detection/detect.py`

Sondes émises par le pipeline slow detect uniquement.
`ncc_match()` n'émet aucune sonde — les métriques NCC
sont portées par `fast_track_thread.py`.
`_build_params()` et le cache kernel ne sont pas instrumentés.

| Nom                            | Type  | Description                       | Conditionnel              |
| ------------------------------ | ----- | --------------------------------- | ------------------------- |
| `detect_slow_ms`               | probe | Durée pipeline slow complet (ms)  | Non                       |
| `detect_slow_candidates_total` | count | Candidats post-filtre géométrique | Non — 0 si aucun candidat |

> `detect_slow_candidates_total` compte les candidats avant remap vers la résolution d'entrée. Les Box filtrées par clamp (`x1 <= x0`) ne sont pas décomptées séparément.
> Décomposition par étape pipeline (`_run_pipeline`) non instrumentée — commentaire `# Bench.timer potentiel part etape` présent dans le code, non implémenté.
> `detect_slow_candidates_total` est émis avant le guard `if not candidates: return []` un appel sans candidat produit bien une émission à 0.

## Documentation associée

- Configuration des canaux + sampling → [`bench-config.md`](bench-config.md)
- Format des lignes JSONL → [`bench-jsonl-schema.md`](bench-jsonl-schema.md)
- Outil d'analyse comparative → [`bench-compare.md`](bench-compare.md)
