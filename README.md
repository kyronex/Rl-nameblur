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

| Sonde                | Type  | Description                               |
| -------------------- | ----- | ----------------------------------------- |
| `capture_wait`       | timer | Attente frame capturée                    |
| `slow_poll`          | timer | Poll résultat DetectThread                |
| `match`              | timer | Matching détections → tracker             |
| `fast_poll`          | timer | Poll résultat FastTrackThread             |
| `predict`            | timer | Tick tracker (predict + TTL + purge)      |
| `blur`               | timer | Application blur + debug overlay          |
| `frames`             | count | Nombre total de frames traitées           |
| `masks_total`        | count | Nombre total de masques confirmés         |
| `fast_wakeup_lag`    | probe | Délai dépôt frame → début traitement (ms) |
| `fast_tick`          | timer | Tick complet FastTrack                    |
| `fast_of_total`      | timer | Phase OF toutes views                     |
| `fast_ncc_total`     | timer | Phase NCC + stale toutes views            |
| `fast_ncc_confirmed` | count | Masques confirmés par NCC                 |
| `fast_stale_used`    | count | Masques servis depuis état stale          |
| `fast_mask_lost`     | count | Masques dépassant `max_stale_frames`      |
| `send`               | timer | Publish frame vers SendThread             |

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
