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

> `main_blur_ms` couvre floutage et overlay debug dans le même `bench.timer`.
> `tracker_confirmed`, `tracker_pending`, `tracker_lost` sont lus dans `main.py`
> via `bench.read_gauge()` pour le log console uniquement — ces gauges sont posées
> dans `tracker.tick()` et documentées dans le domaine `tracker`.
> `motion_staleness_slow_ms` est lu via `bench.last()` pour le log console uniquement —
> posé dans `motion.py` et documenté dans le domaine `motion`.
> `bench.push_frame()` appelé en fin de boucle — infrastructure bench, pas une sonde métier.

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
> la gauge reflète le stock instantané après purge, le count l'historique cumulé
> des transitions avant éventuelle purge.
> `_evict_one()` est appelée avant insertion — `registry_evict_total` est émis
> avant que le nouveau mask entre dans le dict.
> Les gauges d'état sont posées après toutes les transitions du tick — instantané
> cohérent de fin de cycle.
>
> Ces gauges (`registry_confirmed/pending/lost`) sont distinctes des gauges
> `tracker_confirmed/pending/lost` posées dans `tracker.tick()`.
> Les `registry_*` sont destinées au JSONL et aux analyses post-session.
> Les `tracker_*` sont la source de vérité consommée par `main.py` au runtime.)`) comme source de vérité runtime — les deux jeux sont distincts
> et complémentaires.

## Domaine `tracker` — `tracker/tracker.py`

| Sonde                          | Type  | Description                                                  | Conditionnel                 |
| ------------------------------ | ----- | ------------------------------------------------------------ | ---------------------------- |
| `tracker_apply_detections_ms`  | probe | Durée totale de `apply_detections()` (matching slow → masks) | Non                          |
| `tracker_detections_in`        | count | Nombre de détections reçues par `apply_detections()`         | Non — 0 si aucune détection  |
| `tracker_apply_fast_direct_ms` | probe | Durée totale de `apply_fast_direct()` (mise à jour fast)     | Non                          |
| `tracker_fast_drift_skipped`   | count | Masks ignorés pour drift excessif lors du fast update        | Oui — si `drift_skipped > 0` |
| `tracker_tick_ms`              | probe | Durée totale de `tick()` (predict + TTL + purge)             | Non                          |
| `tracker_confirmed`            | gauge | Nombre de masks CONFIRMED (instantané fin de tick)           | Non                          |
| `tracker_pending`              | gauge | Nombre de masks PENDING (instantané fin de tick)             | Non                          |
| `tracker_lost`                 | gauge | Nombre de masks LOST (instantané fin de tick)                | Non                          |
| `tracker_masks_total`          | gauge | Nombre total de masks actifs (tous états)                    | Non                          |

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

> `associator_gated_total` et `associator_candidates_total` sont complémentaires :
> `gated + candidates = N × M` paires totales évaluées.
> `associator_hungarian_rejected_total` agrège deux causes distinctes
> (paire gated et score total < `min_score`) en un seul compteur — non désagrégeable
> sans modification du code d'émission.
> `associator_score_rejected_total` permet d'isoler les rejets score seuls,
> orthogonal à `associator_hungarian_rejected_total`.
> Les counts conditionnels sont absents du JSONL sur les frames
> où leur branche n'est pas atteinte.

## Domaine `motion` — `tracker/motion.py`

Fonctions pures (sans état global). Sondes émises par `apply_detection()` et `predict_position()`.
`compute_predicted_rect()` est intentionnellement sans sonde (appelée N×M fois par l'associator).

| Sonde                      | Type  | Description                                                      | Conditionnel                        |
| -------------------------- | ----- | ---------------------------------------------------------------- | ----------------------------------- |
| `motion_apply_ms`          | probe | Durée de `apply_detection()` (mise à jour état cinématique)      | Non                                 |
| `motion_predict_ms`        | probe | Durée de `predict_position()` (extrapolation position)           | Non                                 |
| `motion_dt_slow_ms`        | probe | Intervalle de temps réel entre deux détections slow (ms)         | Non                                 |
| `motion_staleness_slow_ms` | probe | Staleness accumulée depuis la dernière détection slow (ms)       | Oui — si staleness dépasse le seuil |
| `motion_velocity_pps`      | probe | Vitesse courante du mask (pixels/seconde)                        | Non                                 |
| `motion_residual_px`       | probe | Résidu entre position prédite et position détectée (px)          | Non                                 |
| `motion_teleport_total`    | count | Masks dont le saut de position dépasse le seuil de téléportation | Oui — si saut détecté               |
| `motion_dt_clamped_total`  | count | Masks dont `dt` a été capé par `dt_cap`                          | Oui — si dépassement cap            |
| `motion_staleness_capped`  | count | Masks dont `abs(staleness)` > `dt_cap`                           | Oui — si dépassement cap            |
| `motion_alpha`             | probe | Valeur du facteur de lissage exponentiel appliqué                | Non                                 |
| `motion_predict_dt_ms`     | probe | Intervalle de temps utilisé pour la prédiction (ms)              | Non                                 |
| `motion_predict_shift_px`  | probe | Déplacement prédit appliqué à la position (px)                   | Non                                 |

> `motion_staleness_slow_ms` est lu par `main.py` via `bench.last()` pour le log console.
> `motion_velocity_pps` exprime la vitesse en pixels par seconde —
> ne pas confondre avec `motion_predict_shift_px` qui est un déplacement absolu.
> Les 3 counts conditionnels sont absents du JSONL sur les frames
> où leur branche n'est pas atteinte.

## Domaine `capture` — `threads/capture_thread.py`

| Sonde              | Type  | Description                                        | Conditionnel                                 |
| ------------------ | ----- | -------------------------------------------------- | -------------------------------------------- |
| `capture_frame_ms` | probe | Durée de `source.grab()` (acquisition frame brute) | Non                                          |
| `capture_drop`     | count | Incrémenté si `source.grab()` retourne `None`      | Oui — uniquement si `source.grab()` → `None` |

> `capture_drop` est absent du JSONL sur les frames où `grab()` retourne une frame valide.

---

## Domaine `fast` — `threads/fast_track_thread.py`

Sondes émises par le thread de suivi inter-frames (NCC). Toutes les sondes NCC
sont portées ici — `ncc_match()` dans `detect.py` n'émet aucune sonde.

| Sonde                         | Type  | Description                                             | Conditionnel                 |
| ----------------------------- | ----- | ------------------------------------------------------- | ---------------------------- |
| `fast_track_ms`               | probe | Durée totale du cycle fast track (ms)                   | Non                          |
| `fast_views_total`            | count | Nombre de views traitées par cycle                      | Non — 0 si aucune view       |
| `fast_roi_scale`              | probe | Facteur d'échelle ROI adaptatif par view                | Oui — template présent       |
| `fast_margin_px`              | probe | Valeur margin calculée (pixels) par view                | Oui — template présent       |
| `fast_ncc_score`              | probe | Score NCC par view                                      | Non (0.0 si template absent) |
| `fast_ncc_confirmed_total`    | count | Views confirmées par NCC (score ≥ `ncc_threshold`)      | Oui — NCC réussi             |
| `fast_stale_skipped_total`    | count | Views stale tolérées (`stale ≤ max_stale_frames`)       | Oui — stale toléré           |
| `fast_mask_lost_total`        | count | Views perdues (`stale > max_stale_frames`)              | Oui — stale dépassé          |
| `fast_event_wait_ms`          | probe | Durée d'attente de l'événement déclencheur (ms)         | Non                          |
| `fast_queue_depth`            | gauge | Profondeur de la queue d'entrée au moment du traitement | Non                          |
| `fast_template_missing_total` | count | Views sans template disponible (NCC non tenté)          | Oui — template absent        |
| `fast_roi_too_small_total`    | count | ROI trop petite pour NCC (après scale + margin)         | Oui — ROI dégénérée          |
| `fast_result_drop_total`      | count | Résultats non consommés (queue sortie pleine)           | Oui — queue pleine           |
| `fast_stale_frame_id`         | gauge | `frame_id` de la dernière view stale traitée            | Oui — stale détecté          |
| `fast_ncc_roi_px`             | probe | Surface de la ROI utilisée pour NCC (px²)               | Oui — template présent       |

> Les sondes `fast_roi_scale`, `fast_margin_px`, `fast_ncc_score`, `fast_ncc_roi_px`
> sont émises par view — plusieurs émissions possibles par cycle si plusieurs views actives.
> `fast_ncc_confirmed_total`, `fast_stale_skipped_total`, `fast_mask_lost_total`
> sont des counts cumulatifs session.

## Domaine `mask` — `core/mask.py`

Sondes émises par `Mask.transition()` uniquement.
`to_dict()`, `to_fast_view()`, `__post_init__()` n'émettent aucune sonde.

| Sonde                           | Type  | Description                                                   | Conditionnel                |
| ------------------------------- | ----- | ------------------------------------------------------------- | --------------------------- |
| `mask_transition_matched_total` | count | Transitions déclenchées par un match (détection associée)     | Oui — si transition matched |
| `mask_promote_total`            | count | Transitions PENDING → CONFIRMED                               | Oui — si promotion          |
| `mask_confirm_latency_ms`       | probe | Délai entre création PENDING et promotion CONFIRMED (ms)      | Oui — si promotion          |
| `mask_revive_total`             | count | Transitions LOST → CONFIRMED (revive)                         | Oui — si revive détecté     |
| `mask_revive_latency_ms`        | probe | Délai entre entrée LOST et revive (ms)                        | Oui — si revive détecté     |
| `mask_transition_missing_total` | count | Transitions déclenchées sans match (mask non matché ce cycle) | Oui — si transition missing |
| `mask_to_lost_total`            | count | Transitions vers état LOST                                    | Oui — si transition → LOST  |
| `mask_lost_latency_ms`          | probe | Délai entre dernière détection et transition LOST (ms)        | Oui — si transition → LOST  |

> `mask_confirm_latency_ms` et `mask_revive_latency_ms` mesurent des délais
> calculés à l'instant de la transition — non rétroactifs.
> `mask_transition_matched_total` et `mask_transition_missing_total` sont
> complémentaires : chaque appel à `transition()` incrémente l'un ou l'autre.
> Les sondes conditionnelles sont absentes du JSONL sur les frames
> sans transition correspondante.

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
>
> `bench.count()` n'est émis que sur succès — les tentatives échouées
> ne produisent aucune sonde. Les tentatives échouées sont tracées en log uniquement.

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
>
> `detect_slow_candidates_total` est émis avant le guard `if not candidates: return []` —
> un appel sans candidat produit bien une émission à 0.

## Documentation associée

- Configuration des canaux + sampling → [`bench-config.md`](bench-config.md)
- Format des lignes JSONL → [`bench-jsonl-schema.md`](bench-jsonl-schema.md)
- Outil d'analyse comparative → [`bench-compare.md`](bench-compare.md)
