# RL-NameBlur

Script Python pour anonymiser en temps réel les noms des joueurs dans Rocket League.
Capture le flux vidéo, détecte les cartouches de noms (HSV), floute les zones et renvoie
le flux vers OBS via une caméra virtuelle.

---

## Architecture du pipeline

```text
SourceSelector (probe DXCam → cv2 fallback)
    │
    ▼
CaptureThread (worker daemon)
    ├─ _latest_frame (RGB, copy)
    ├─ _latest_ts   (perf_counter)
    └─ _frame_id    (monotone, sert au cache FastTrack)
    │
    ▼ get_frame() / get_frame_id() — non bloquant, lock court
    │
    ├──→ DetectThread (Slow — full frame)
    │        ├─ give_frame(frame, ts) ← push depuis main loop
    │        └─ get_result() → (zones, zones_ts, count, frame_ts_detected)
    │
    ├──→ FastTrackThread (Fast — Event-driven, OF→NCC→stale fallback)
    │        ├─ give_frame_and_views(frame, views, ts) ← push depuis main loop
    │        └─ get_results() → (version, [(uid, rect, score)], ts)
    │
    ├──→ Main Loop ──→ Tracker (TrackerConfig)
    │        ├─ tick(detections, ts)
    │        │     ├─ Associator — gating + coût IoU+pHash + Hungarian
    │        │     │     └─ matches / unmatched_dets / unmatched_masks
    │        │     ├─ MaskRegistry — CRUD + TTL + éviction
    │        │     │     └─ create / update / expire masks
    │        │     └─ → confirmed_masks (MaskState)
    │        └─ get_confirmed_masks()
    │
    ├──→ core/blur.py (apply_blur — pixelate / box / gaussian / fill — in-place)
    │
    └──→ SendThread (double buffer zéro-copie, swap pointeurs)
             ├─ borrow() → write buffer (main écrit in-place)
             ├─ publish() → swap write/send buffers
             └─ worker → vcam.send(send_buf) (OBS Virtual Camera)
```

---

## Fichiers

### Entrée / Orchestration

| Fichier   | Rôle                                                                               |
| --------- | ---------------------------------------------------------------------------------- |
| `main.py` | Orchestration pipeline v13 — TTL, matching, dual detect, Tracker, MaskState, debug |

### Configuration

| Fichier       | Rôle                                                         |
| ------------- | ------------------------------------------------------------ |
| `config.py`   | Singleton config — charge config.yaml + hot-reload (watcher) |
| `config.yaml` | Paramètres centralisés (zéro magic number)                   |

### Capture

| Fichier                   | Rôle                                                                   |
| ------------------------- | ---------------------------------------------------------------------- |
| `capture/base.py`         | Interface `CaptureSource` + exception `CaptureSourceNotFound`          |
| `capture/config.py`       | Snapshot immuable `CaptureConfig` (paramètres `capture.*`)             |
| `capture/selector.py`     | `SourceSelector.resolve()` — probe et sélection automatique du backend |
| `capture/dxcam_source.py` | Backend DXCam (Windows, DirectX — prioritaire)                         |
| `capture/mss_source.py`   | Backend MMS                                                            |
| `capture/cv2_source.py`   | Backend OpenCV VideoCapture (fallback OBS Virtual Camera)              |
| `capture/wgc_source.py`   | Backend Windows Graphics Capture (WGC)                                 |

#### Paramètres clés à tuner

> Le fichier `config.yaml` est intégralement commenté et constitue la référence principale.
> Les paramètres ci-dessous sont ceux qui impactent le plus le comportement du pipeline.

| Clé YAML                       | Défaut                           | Description                                                 |
| ------------------------------ | -------------------------------- | ----------------------------------------------------------- |
| `screen.capture_fps`           | `120`                            | FPS cible de capture DXCam                                  |
| `screen.vcam_fps`              | `120`                            | FPS cible de sortie vers OBS Virtual Camera                 |
| `capture.source_priority`      | `["wgc", "dxcam", "cv2", "mss"]` | Ordre de tentative des backends de capture                  |
| `capture.probe_timeout_s`      | `0.5`                            | Timeout (s) par probe source au démarrage                   |
| `capture.probe_min_frames`     | `1`                              | Frames min reçues pour valider un backend                   |
| `capture.cv2.device_name`      | `"OBS"`                          | Substring du nom DirectShow pour la source cv2              |
| `capture.mss.monitor_index`    | `1`                              | Ecran selectionner comme source mms                         |
| `capture.wgc.target`           | `["monitor","window"]`           | Type de source utiliser pour wgc                            |
| `capture.wgc.monitor_index`    | `1`                              | Ecran selectionner comme source wgc                         |
| `capture.wgc.window_name`      | `"Rocket League"`                | Substring du nom Exec. pour la source wgc                   |
| `detect.slow.scale`            | `2.0`                            | Facteur de downscale pour la détection HSV (coût CPU)       |
| `detect.fast.enabled`          | `true`                           | Active le FastTracker inter-frames (OF → NCC → stale)       |
| `detect.fast.ncc_threshold`    | `0.4`                            | Seuil NCC pour valider un match template → ROI (∈ [0, 1])   |
| `detect.fast.max_stale_frames` | `15`                             | Frames consécutives sans match NCC avant perte du masque    |
| `blur.mode`                    | `pixelate`                       | Mode de floutage : `pixelate` / `box` / `gaussian` / `fill` |
| `blur.pixel_size`              | `15`                             | Taille d'un bloc (px) en mode `pixelate`                    |
| `debug.overlay.enabled`        | `false`                          | Affiche les rectangles et UIDs sur le flux OBS              |
| `debug.bench.enabled`          | `true`                           | Active la collecte des métriques de performance             |

---

## Métriques bench

Instrumentation centralisée via `core/bench.py`.

Quatre types de sondes :

- `timer` — context manager mesurant un bloc, écrit via `probe()` en sortie (`with bench.timer(name)`)
- `probe` — valeur scalaire échantillonnée (durée, score, aire…)
- `count` — compteur cumulatif incrémental
- `gauge` — valeur instantanée écrasée à chaque mesure

### Configuration (`config.yaml`)

> Schéma complet des fichiers JSONL produits : [`docs/bench-jsonl-schema.md`](docs/bench-jsonl-schema.md).

```yaml
debug:
  bench:
    enabled: true # Active BenchRegistry + démarre les writers si writer.enabled=true
    history_window_s: 60 # Fenêtre glissante conservée en mémoire pour summary_window() (s)

    writer:
      enabled: true # Maître : false = aucun writer démarré
      queue_maxsize: 10000 # Drop + bench.count("bench_writer_dropped") au-delà
      shutdown_timeout_s: 2.0 # Délai max accordé à chaque writer pour vider sa queue
      session_id_format: "%Y%m%d_%H%M%S" # Inséré dans le nom de fichier avant l'extension

    agg:
      enabled: true
      path: "logs/json/bench_agg.jsonl" # → bench_agg_{session_id}.jsonl
      interval_s: 1.0

    frame:
      enabled: true
      path: "logs/json/bench_frame.jsonl" # → bench_frame_{session_id}.jsonl

    fast:
      enabled: true
      path: "logs/json/bench_fast.jsonl" # → bench_fast_{session_id}.jsonl
      interval_s: 1.0
```

**Hiérarchie d'activation** :

- `debug.bench.enabled: false` → `BenchRegistry` désactivé, aucun writer démarré, toutes les sondes sont des no-ops.
- `debug.bench.writer.enabled: false` → sondes actives en mémoire, aucun fichier écrit.
- Chaque canal (`agg` / `frame` / `fast`) peut être désactivé indépendamment via son propre `enabled`.

**Canaux JSONL** :

| Canal   | Fichier                          | Cadence                     |
| ------- | -------------------------------- | --------------------------- |
| `frame` | `bench_frame_{session_id}.jsonl` | 1 ligne / frame capturée    |
| `agg`   | `bench_agg_{session_id}.jsonl`   | 1 ligne / `agg.interval_s`  |
| `fast`  | `bench_fast_{session_id}.jsonl`  | 1 ligne / `fast.interval_s` |

> Voir [`docs/bench-jsonl-schema.md`](docs/bench-jsonl-schema.md) pour la structure exacte de chaque ligne.

---

### Sondes par fichier

#### `main.py` — boucle principale

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

### Domaine `registry` — `tracker/registry.py`

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

### Domaine `tracker` — `tracker/tracker.py`

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

### Domaine `associator` — `tracker/associator.py`

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

### Domaine `motion` — `tracker/motion.py`

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

### Domaine `fast` — `threads/fast_track_thread.py`

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

### Domaine `mask` — `core/mask.py`

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

### Domaine `capture` — `threads/capture_thread.py`

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

### Domaine `selector` — `capture/selector.py`

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

### Domaine `detect` — `detection/detect.py`

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

## Dépendances

```bash
pip install opencv-python numpy pyvirtualcam dxcam pyyaml scipy pandas screeninfo pygrabber mss windows-capture
```

| Lib               | Version   | Usage                                           |
| ----------------- | --------- | ----------------------------------------------- |
| `opencv-python`   | 4.13.0.92 | Traitement image (HSV, blur, OF, NCC)           |
| `numpy`           | 2.4.2     | Buffers, opérations vectorielles                |
| `pyvirtualcam`    | 0.15.0    | Envoi vers OBS Virtual Camera                   |
| `dxcam`           | 0.0.5     | Capture écran DirectX (Windows)                 |
| `pyyaml`          | 6.0.3     | Chargement config.yaml                          |
| `scipy`           | 1.17.1    | Assignation hongroise (`linear_sum_assignment`) |
| `pandas`          | 3.0.2     | Export CSV métriques bench                      |
| `screeninfo`      | 0.8.1     | Détection résolution moniteur                   |
| `comtypes`        | 1.4.15    | Dépendance interne dxcam / pyvirtualcam         |
| `pygrabber`       | 0.2       | Énumération DirectShow par nom (capture cv2)    |
| `mss`             | 10.2.0    | Capture écran multi-plateforme (fallback DXCam) |
| `windows-capture` | 2.0.0     | Capture écran multi-plateforme (fallback DXCam) |

---

## Lancement

**Prérequis source** : au moins une des deux conditions doit être remplie au démarrage :

- Rocket League lancé en mode **Borderless Window** ou **Fullscreen** (backend DXCam)
- OBS ouvert avec une source **Game Capture** et la **caméra virtuelle démarrée**
  (`Outils → Démarrer la caméra virtuelle`) (backend cv2)

`SourceSelector` sonde automatiquement les backends dans l'ordre `capture.source_priority`
et retient le premier opérationnel. Si aucun ne répond, le script s'arrête avec un message
d'erreur actionnable.

```bash
python main.py
```

Ctrl+C pour arrêter proprement — le benchmark complet s'affiche à la sortie.

> **Note** : la bascule DXCam → cv2 n'est pas gérée à chaud après démarrage.
> Si la source active disparaît en cours de session, relancez le script.
