# Sondes bench par fichier

[← Retour au README](./../README.md)

Référentiel exhaustif des sondes émises par chaque module instrumenté de l'application RL-NameBlur. Chaque section couvre un fichier ou un domaine fonctionnel.

## Sommaire

- [`main.py` — boucle principale](#mainpy--boucle-principale)
- [Domaine `registry` — `tracker/registry.py`](#domaine-registry--trackerregistrypy)
- [Domaine `tracker` — `tracker/tracker.py`](#domaine-tracker--trackertrackerpy)
- [Domaine `mask` — `core/mask.py`](#domaine-mask--coremaskpy)
- [Domaine `capture` — `threads/capture_thread.py`](#domaine-capture--threadscapture_threadpy)
- [Domaine `selector` — `capture/selector.py`](#domaine-selector--captureselectorpy)
- [Domaine `detect` — `detection/detect.py`](#domaine-detect--detectiondetectpy)

---

## `main.py` — boucle principale

| Sonde                  | Type  | Description                                           | Conditionnel                                 |
| ---------------------- | ----- | ----------------------------------------------------- | -------------------------------------------- |
| `main_capture_wait_ms` | probe | Durée attente frame capturée (`capturer.get_frame()`) | Non                                          |
| `main_frame_id`        | gauge | Identifiant de la frame courante                      | Non                                          |
| `main_slow_poll_ms`    | probe | Durée poll résultat `DetectThread`                    | Non                                          |
| `main_match_ms`        | probe | Durée matching détections slow → tracker              | Oui — si `slow_updated and new_plates`       |
| `main_fast_poll_ms`    | probe | Durée poll résultat `FastTrackThread`                 | Oui — si `fast_enabled and not slow_updated` |
| `main_predict_ms`      | probe | Durée `tracker.tick()` (predict + TTL + purge)        | Non                                          |
| `main_blur_ms`         | probe | Durée floutage + overlay debug                        | Non                                          |
| `main_send_ms`         | probe | Durée publication frame vers `SendThread`             | Non                                          |
| `main_frames_total`    | count | Nombre cumulé de frames traitées                      | Non                                          |
| `main_masks_total`     | gauge | Nombre de masques confirmés à la frame courante       | Non                                          |

> `main_match_ms` et `main_fast_poll_ms` sont conditionnels — absents du JSONL sur les frames
> où leur branche n'est pas exécutée.
> `bench.push_frame()` est appelé **chaque frame** en fin de boucle, déclenchant le flush canal `frame`.

## Domaine `registry` — `tracker/registry.py`

Sondes émises par `MaskRegistry`. Les gauges sont recalculées
à chaque appel `tick_and_expire()`, en fin de boucle après
toutes les transitions.

| Nom                     | Type  | Description                               | Conditionnel                                      |
| ----------------------- | ----- | ----------------------------------------- | ------------------------------------------------- |
| `registry_create_total` | count | Incrémenté à chaque création de `Mask`    | Non                                               |
| `registry_lost_total`   | count | Incrémenté à chaque transition → LOST     | Oui — si mask non matché depuis > `lost_after_s`  |
| `registry_expire_total` | count | Incrémenté à chaque purge de mask LOST    | Oui — si mask LOST depuis > `expire_after_lost_s` |
| `registry_evict_total`  | count | Incrémenté à chaque éviction capacité max | Oui — si `len(masks) >= max_masks`                |
| `registry_confirmed`    | gauge | Nombre de masks en état CONFIRMED         | Non                                               |
| `registry_pending`      | gauge | Nombre de masks en état PENDING           | Non                                               |
| `registry_lost`         | gauge | Nombre de masks en état LOST              | Non                                               |

> Les 4 counts (`_total`) sont cumulatifs depuis le démarrage de la session.
> `registry_lost` (gauge) et `registry_lost_total` (count) sont orthogonaux :
> la gauge reflète le stock instantané, le count l'historique cumulé des transitions.

## Domaine `tracker` — `tracker/tracker.py`

Sondes émises par `Tracker`. Les gauges sont recalculées en fin
de `tick()`, après que `registry.tick_and_expire()` a effectué
toutes les transitions TTL.

| Nom                            | Type  | Description                                          | Conditionnel               |
| ------------------------------ | ----- | ---------------------------------------------------- | -------------------------- |
| `tracker_apply_detections_ms`  | probe | Durée totale `apply_detections()`                    | Non                        |
| `tracker_detections_in`        | count | Nombre de détections reçues par `apply_detections()` | Non                        |
| `tracker_apply_fast_direct_ms` | probe | Durée totale `apply_fast_direct()`                   | Non                        |
| `tracker_fast_drift_skipped`   | count | Masks fast skippés (drift > `fast_max_drift_s`)      | Oui — si drift_skipped > 0 |
| `tracker_tick_ms`              | probe | Durée totale `tick()` (predict + TTL + purge)        | Non                        |
| `tracker_confirmed`            | gauge | Nombre de masks CONFIRMED post-purge                 | Non                        |
| `tracker_pending`              | gauge | Nombre de masks PENDING post-purge                   | Non                        |
| `tracker_lost`                 | gauge | Nombre de masks LOST post-purge                      | Non                        |
| `tracker_masks_total`          | gauge | Total masks (CONFIRMED + PENDING + LOST) post-purge  | Non                        |

> `tracker_confirmed / pending / lost` coexistent avec `registry_confirmed / pending / lost` :
> les variantes `registry_*` sont émises dans `registry.tick_and_expire()`,
> les variantes `tracker_*` sont émises dans `tracker.tick()` après retour de la registry.
> `main.py` consomme exclusivement les variantes `tracker_*`.
> `tracker_fast_drift_skipped` est conditionnel — absent du JSONL sur les frames
> où aucun mask n'est skippé.

## Domaine `associator` — `tracker/associator.py`

Sondes émises par `Associator`. Les counts `_build_cost_matrix` sont
émis **par paire** (det × mask) — leur valeur absolue dépend de N×M.

| Nom                                   | Type  | Description                                                 | Conditionnel               |
| ------------------------------------- | ----- | ----------------------------------------------------------- | -------------------------- |
| `associator_gated_total`              | count | Paires rejetées par gate géographique                       | Oui — par paire hors rayon |
| `associator_candidates_total`         | count | Paires ayant passé la gate géographique                     | Oui — par paire dans rayon |
| `associator_score_rejected_total`     | count | Paires rejetées (score < min_score) avant Hungarian         | Oui — par paire sous seuil |
| `associator_tick_ms`                  | probe | Durée totale `associate()`                                  | Non                        |
| `associator_hungarian_rejected_total` | count | Assignations rejetées post-Hungarian (gated ou score < min) | Oui — si rejet             |
| `associator_matched_total`            | count | Matches retenus après Hungarian                             | Non                        |
| `associator_unmatched_det_total`      | count | Détections sans mask associé                                | Non                        |
| `associator_unmatched_mask_total`     | count | Masks sans détection associée                               | Non                        |

> `associator_gated_total` et `associator_candidates_total` sont complémentaires :
> `gated + candidates = N × M` paires totales évaluées.
> `associator_hungarian_rejected_total` agrège deux causes distinctes
> (paire gated et score total < `min_score`) en un seul compteur — non désagrégeable
> sans modification du code d'émission.
> Les 6 counts conditionnels sont absents du JSONL sur les frames
> où leur branche n'est pas atteinte.

## Domaine `motion` — `tracker/motion.py`

Fonctions pures (sans état global). Sondes émises par deux fonctions publiques : `apply_detection()` et `predict_position()`.
`compute_predicted_rect()` est intentionnellement sans sonde (appelée N×M fois par l'associator).

| Nom                        | Type  | Description                                       | Conditionnel                                  |
| -------------------------- | ----- | ------------------------------------------------- | --------------------------------------------- |
| `motion_apply_ms`          | probe | Durée totale `apply_detection()`                  | Non                                           |
| `motion_residual_px`       | probe | Distance centre prédit vs observé avant mutation  | Oui — source=slow ET last_slow_ts > 0         |
| `motion_dt_slow_ms`        | probe | Délai entre deux détections slow consécutives     | Oui — source=slow ET last_slow_ts > 0         |
| `motion_dt_clamped_total`  | count | Détections slow rejetées (dt > dt_slow_max)       | Oui — si dt excessif                          |
| `motion_teleport_total`    | count | Détections slow rejetées (dist > teleport_thresh) | Oui — si saut spatial                         |
| `motion_velocity_pps`      | probe | Norme vitesse EMA post-update                     | Oui — source=slow uniquement, toutes branches |
| `motion_predict_ms`        | probe | Durée totale `predict_position()`                 | Non                                           |
| `motion_staleness_slow_ms` | probe | Délai entre last_slow_ts et now (fraîcheur slow)  | Oui — last_slow_ts > 0                        |
| `motion_staleness_capped`  | count | Masks dont abs(staleness) > dt_cap                | Oui — si dépassement cap                      |

> `motion_staleness_slow_ms` est alimentée **uniquement** depuis
> `predict_position()` (1×/mask non matché/tick), jamais depuis
> `compute_predicted_rect()`. Un mask suivi correctement par le fast
> tracker peut afficher une staleness slow élevée sans anomalie.

## Domaine `fast` — `threads/fast_track_thread.py`

Sondes émises par le worker interne de `FastTrackThread`.
Les sondes `fast_margin_ms`, `fast_margin_px`, `fast_ncc_score`
sont émises **par view** (N fois par tick si N views actives).

| Nom                         | Type  | Description                                      | Conditionnel                 |
| --------------------------- | ----- | ------------------------------------------------ | ---------------------------- |
| `fast_wakeup_lag_ms`        | probe | Lag entre dépôt frame et début traitement worker | Non                          |
| `fast_tick_total`           | count | Ticks traités par le worker                      | Non                          |
| `fast_n_masks`              | probe | Nombre de views reçues par tick                  | Non                          |
| `fast_tick_ms`              | probe | Durée totale du tick (OF + NCC + publish)        | Non                          |
| `fast_cvt_ms`               | probe | Durée conversion RGB→gray                        | Non                          |
| `fast_of_total_ms`          | probe | Durée phase OF (toutes views)                    | Non                          |
| `fast_mask_processed_total` | count | Views traitées en phase OF                       | Non                          |
| `fast_of_failed_total`      | count | Échecs OF par view                               | Oui — si OF échoué           |
| `fast_ncc_total_ms`         | probe | Durée phase NCC (toutes views)                   | Non                          |
| `fast_margin_ms`            | probe | Durée calcul margin adaptative par view          | Oui — template présent       |
| `fast_margin_px`            | probe | Valeur margin calculée (pixels) par view         | Oui — template présent       |
| `fast_ncc_score`            | probe | Score NCC par view                               | Non (0.0 si template absent) |
| `fast_ncc_confirmed_total`  | count | Views confirmées par NCC                         | Oui — NCC réussi             |
| `fast_stale_skipped_total`  | count | Views stale tolérées (stale ≤ max_stale_frames)  | Oui — stale toléré           |
| `fast_mask_lost_total`      | count | Views perdues (stale > max_stale_frames)         | Oui — stale dépassé          |

> Les anciennes sondes `fast_ncc_ok`, `fast_stale_skipped`, `fast_ncc_roi_too_small`
> (README pré-audit) sont **supprimées** — remplacées par ce bloc.
> La section « Sondes restantes à localiser ⏳ » est **retirée**
> pour `fast_track_thread.py` — audit complet livré ici.

## Domaine `mask` — `core/mask.py`

Sondes émises par `Mask.transition()` uniquement.
`to_dict()`, `to_fast_view()`, `__post_init__()` n'émettent aucune sonde.

| Nom                             | Type  | Description                                    | Conditionnel                         |
| ------------------------------- | ----- | ---------------------------------------------- | ------------------------------------ |
| `mask_transition_matched_total` | count | Appels `transition("matched")`                 | Non                                  |
| `mask_promote_total`            | count | Transitions PENDING→CONFIRMED                  | Oui — frames_matched ≥ confirm_after |
| `mask_confirm_latency_ms`       | probe | Délai created_ts→CONFIRMED (ms)                | Oui — même condition que promote     |
| `mask_revive_total`             | count | Transitions LOST→CONFIRMED                     | Oui — state == LOST sur matched      |
| `mask_revive_latency_ms`        | probe | Durée LOST→CONFIRMED depuis lost_since_ts (ms) | Oui — prev_lost_since_ts not None    |
| `mask_transition_missing_total` | count | Appels `transition("missing")`                 | Non                                  |
| `mask_to_lost_total`            | count | Transitions PENDING/CONFIRMED→LOST             | Oui — since_last_seen ≥ lost_after_s |
| `mask_lost_latency_ms`          | probe | Durée created_ts→LOST (ms)                     | Oui — même condition que to_lost     |

> `mask_transition_matched_total` et `mask_transition_missing_total`
> comptent les **appels** à `transition()`, pas les transitions effectives.
> Un appel `transition("missing")` sans franchissement de seuil
> incrémente le compteur sans émettre `mask_to_lost_total`.
>
> `mask_lost_latency_ms` mesure la durée de vie totale jusqu'à la perte
> (`ts - created_ts`), pas le délai depuis le dernier match
> (`ts - last_seen_ts`).

## Domaine `capture` — `threads/capture_thread.py`

Sondes émises par le worker interne de `CaptureThread`.
`get_frame()` et `get_frame_id()` n'émettent aucune sonde —
la mesure d'attente côté consommateur est dans `main.py`
(`main_capture_wait_ms`).

| Nom                | Type  | Description                          | Conditionnel        |
| ------------------ | ----- | ------------------------------------ | ------------------- |
| `capture_frame_ms` | probe | Durée `source.grab()` (ms)           | Non                 |
| `capture_drop`     | count | Frames None retournées par la source | Oui — grab() → None |

> `capture_frame_ms` ne couvre pas `frame.copy()` ni l'acquisition
> du lock — coût de copie non instrumenté.
>
> FPS réel de capture est loggué toutes les 5 s (`[CaptureThread] FPS réel`)
> mais **non exposé dans bench**. Aucune sonde `capture_fps` n'existe
> dans ce fichier.

---

## Domaine `selector` — `capture/selector.py`

Sonde émise une seule fois par session à la sélection de source.
`_probe()` n'émet aucune sonde bench — les tentatives échouées
sont tracées en log uniquement.

| Nom                      | Type  | Description                                | Conditionnel           |
| ------------------------ | ----- | ------------------------------------------ | ---------------------- |
| `selector_source_<name>` | count | Source retenue (`dxcam`/`cv2`/`mss`/`wgc`) | Oui — resolve() réussi |

> Famille dynamique : `<name>` est le nom littéral de la source retenue.
> Zéro émission si `CaptureSourceNotFound` est levée.
>
> Écart vs Plan_Bench.md : Plan spécifiait `gauge`, implémenté en `count`.
> Décision définitive : `count` retenu (émission unique par session,
> la sémantique gauge n'apporte rien ici).

## Domaine `detect` — `detection/detect.py`

Sondes émises par le pipeline slow detect uniquement.
`ncc_match()` n'émet aucune sonde — les métriques NCC
sont portées par `fast_track_thread.py`.
`_build_params()` et le cache kernel ne sont pas instrumentés.

| Nom                            | Type  | Description                       | Conditionnel              |
| ------------------------------ | ----- | --------------------------------- | ------------------------- |
| `detect_slow_ms`               | probe | Durée pipeline slow complet (ms)  | Non                       |
| `detect_slow_candidates_total` | count | Candidats post-filtre géométrique | Non — 0 si aucun candidat |

> `detect_slow_candidates_total` compte les candidats avant remap
> vers la résolution d'entrée. Les Box filtrées par clamp
> (`x1 <= x0`) ne sont pas décomptées séparément.
>
> Décomposition par étape pipeline (`_run_pipeline`) non instrumentée —
> commentaire `# Bench.timer potentiel part etape` présent dans le code,
> non implémenté.

---

## Documentation associée

- Configuration des canaux + sampling → [`bench-config.md`](bench-config.md)
- Format des lignes JSONL → [`bench-jsonl-schema.md`](bench-jsonl-schema.md)
- Outil d'analyse comparative → [`bench-compare.md`](bench-compare.md)
