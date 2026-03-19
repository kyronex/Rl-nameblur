# RL-NameBlur

Script Python pour anonymiser en temps réel les noms des joueurs dans Rocket League.
Capture le flux vidéo, détecte les cartouches de noms (HSV), floute les zones et renvoie
le flux vers OBS via une caméra virtuelle.

---

## Architecture du pipeline

```text
DXCam (120fps)
    │
    ▼
CaptureThread ──→ frame RGB
    │
    ├──→ DetectThread (Slow — , full frame)
    │        └──→ zones [(x, y, w, h), ...]
    │
    ├──→ FastTrackThread (Fast — ROI redetect autour des masques connus)
    │        └──→ zones mises à jour [(x, y, w, h), ...]
    │
    ├──→ Main Loop (TTL + matching distance/IoU + smooth + prédiction vélocité)
    │        └──→ active_masks → blur_zones
    │
    ├──→ blur.py (pixelate / box / gaussian / fill — in-place)
    │
    └──→ SendThread ──→ OBS Virtual Camera (pyvirtualcam)
```

---

## Fichiers

| Fichier                 | Rôle                                                                         |
| ----------------------- | ---------------------------------------------------------------------------- |
| `main.py`               | Orchestration pipeline v10, TTL, matching, dual detect, CSV benchmark, debug |
| `config.py`             | Singleton config — charge config.yaml + hot-reload (watcher)                 |
| `config.yaml`           | Paramètres centralisés (zéro magic number)                                   |
| `capture_thread.py`     | Thread de capture DXCam non-bloquant                                         |
| `detect_thread.py`      | Thread de détection lente (slow — full frame, scale lent)                    |
| `fast_track_thread.py`  | Thread de tracking rapide (fast — ROI autour des masques)                    |
| `detect.py`             | Pipeline HSV dual-pass ( white masking)                                      |
| `detect_tools.py`       | Utilitaires dessin (write_circles, write_rects, get_color)                   |
| `detect_tools_mask.py`  | Masques image (saturation, white mask, sobel, refine/merge)                  |
| `detect_tools_boxes.py` | Boîtes englobantes (extract, split, validate, merge, process_channel)        |
| `detect_stats.py`       | Compteurs de performance thread-safe (timings + rejections)                  |
| `blur.py`               | Floutage multi-mode (pixelate, box, gaussian, fill)                          |
| `send_thread.py`        | Thread d'envoi vers la caméra virtuelle OBS                                  |
| `bench_pipeline.py`     | Benchmark pas-à-pas du pipeline detect sur une image                         |

---

## Paramètres principaux (`config.yaml`)

| Clé YAML                 | Valeur par défaut | Description                                                  |
| ------------------------ | ----------------- | ------------------------------------------------------------ |
| `screen.width`           | 1920              | Résolution capture                                           |
| `screen.height`          | 1080              | Résolution capture                                           |
| `screen.capture_fps`     | 120               | FPS cible DXCam                                              |
| `screen.vcam_fps`        | 120               | FPS déclaré à OBS                                            |
| `masks.ttl_max`          | 4                 | Persistance d'un masque (en frames)                          |
| `masks.max_masks`        | 8                 | Nombre max de masques actifs simultanément                   |
| `masks.smooth_alpha`     | 1.0               | Lissage exponentiel des rects (0=fixe, 1=immédiat)           |
| `matching.mode`          | `"distance"`      | Algo de matching : `"distance"` ou `"iou"`                   |
| `matching.dist_thresh`   | 60                | Seuil distance (px) pour matcher un masque                   |
| `matching.iou_thresh`    | 0.15              | Seuil IoU pour matcher un masque                             |
| `blur.mode`              | `"fill"`          | Mode de flou : `"pixelate"`, `"box"`, `"gaussian"`, `"fill"` |
| `blur.pixel_size`        | 13                | Taille bloc pixelate                                         |
| `blur.strength`          | 27                | Force du flou gaussien / box                                 |
| `blur.fill_color`        | `[80, 80, 80]`    | Couleur de remplissage (mode fill)                           |
| `blur.margin`            | 0                 | Marge (px) autour de chaque zone floutée                     |
| `detect.slow.scale`      | 2.0               | Facteur de réduction (slow detect)                           |
| `detect.fast.scale`      | 3.0               | Facteur de réduction (fast ROI tracking)                     |
| `detect.fast.roi_margin` | 0.6               | Marge ROI autour des masques existants (+%)                  |
| `detect.fast.enabled`    | true              | Active/désactive le fast tracking                            |
| `debug.log_level`        | `"WARNING"`       | Niveau de log : DEBUG, INFO, WARNING, ERROR                  |
| `debug.draw`             | false             | Affiche les rectangles colorés TTL dans OBS                  |
| `debug.colors.fresh`     | `[0, 255, 0]`     | Couleur masque TTL élevé (vert)                              |
| `debug.colors.persist`   | `[0, 255, 255]`   | Couleur masque TTL moyen (jaune)                             |
| `debug.colors.dying`     | `[0, 0, 255]`     | Couleur masque TTL bas (rouge)                               |

---

## Features livrées

```text
══════════════════════════════════════════════════════════════════════════
 #  │ Feature                          │ Version │ Impact
════╪══════════════════════════════════╪═════════╪════════════════════════
                    CAPTURE
────┼──────────────────────────────────┼─────────┼────────────────────────
 1  │ Capture écran DXCam              │ v1.0    │ 120fps cible Windows
 2  │ CaptureThread (non-bloquant)     │ v5.0    │ Capture découplée
────┼──────────────────────────────────┼─────────┼────────────────────────
                    DÉTECTION
────┼──────────────────────────────────┼─────────┼────────────────────────
 3  │ Détection HSV (orange + bleu)    │ v1.0    │ Repère les cartouches
 4  │ Dual-pass HSV (v2)               │ v3.0    │ 2 passes séparées
 5  │ White masking (core + ext)       │ v8.0    │ Filtre texte blanc
 6  │ Morphologie (open + close)       │ v2.0    │ Nettoyage contours
 7  │ Filtres de forme (ratio, area)   │ v2.0    │ Élimine gros blocs
 8  │ DetectThread slow (non-bloquant) │ v5.0    │ Detect découplé
 9  │ Resize avant detect (SCALE)      │ v5.0    │ Accélère le HSV
10  │ FastTrackThread (ROI redetect)   │ v8.0    │ Tracking rapide ROI
────┼──────────────────────────────────┼─────────┼────────────────────────
                    SUIVI / MASQUES
────┼──────────────────────────────────┼─────────┼────────────────────────
11  │ Système TTL (persistance)        │ v4.0    │ Masque survit N frames
12  │ Matching par distance            │ v4.0    │ Réidentifie un masque
13  │ Matching par IoU                 │ v4.0    │ Alternative au distance
14  │ MAX_MASKS (cap masques actifs)   │ v4.0    │ Limite les faux positifs
15  │ Smooth exponentiel (rects)       │ v5.0    │ Réduit les sauts
16  │ Prédiction par vélocité          │ v8.0    │ Anticipe le déplacement
────┼──────────────────────────────────┼─────────┼────────────────────────
                    FLOUTAGE
────┼──────────────────────────────────┼─────────┼────────────────────────
17  │ Pixelate in-place                │ v3.0    │ Flou rapide et discret
18  │ Multi-mode blur                  │ v8.0    │ pixelate/box/gaussian/fill
19  │ Marge configurable               │ v5.0    │ Couvre débordements
────┼──────────────────────────────────┼─────────┼────────────────────────
                    SORTIE
────┼──────────────────────────────────┼─────────┼────────────────────────
20  │ SendThread (pyvirtualcam)        │ v5.0    │ Envoi OBS non-bloquant
21  │ Buffer zéro-copie (borrow/pub)   │ v5.1    │ Évite allocations
────┼──────────────────────────────────┼─────────┼────────────────────────
                    CONFIG / INFRA
────┼──────────────────────────────────┼─────────┼────────────────────────
22  │ config.yaml centralisé           │ v6.1    │ Zéro magic number
23  │ config.py singleton              │ v6.1    │ Import unique partout
24  │ Hot-reload config (watcher)      │ v8.0    │ Modif YAML à chaud
25  │ DEBUG_DRAW (rects colorés TTL)   │ v5.0    │ Visualise les masques
26  │ Benchmark intégré (tous modules) │ v5.0    │ Stats à la sortie
27  │ detect_stats thread-safe         │ v8.0    │ Compteurs par pipeline
════╪══════════════════════════════════╪═════════╪════════════════════════

```

---

## Performances mesurées (v6.1 — i7-12700K)

| Étape           | Temps moyen |
| --------------- | ----------- |
| Capture DXCam   | ~28 ms      |
| Détection HSV   | ~102 ms     |
| Blur + CVT      | ~3 ms       |
| Envoi vcam      | ~13 ms      |
| Loop principale | ~20 ms      |
| **FPS observé** | **~49 FPS** |

### Corrélation masques / FPS

| Masques actifs | FPS observé |
| -------------- | ----------- |
| 0              | ~57 FPS     |
| 3–4            | ~50 FPS     |
| 6–8            | ~44 FPS     |

> Chaque masque coûte environ **~2 FPS**. Réduire les faux positifs = gain direct.

---

## Plan de développement

```txt
══════════════════════════════════════════════════════════════════════════════════════════
 #  │ Feature                               │ Statut     │ Notes
════╪═══════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 4 — QUALITÉ DÉTECTION
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
28  │ Score de confiance par masque         │ ⬚ À FAIRE │ Prioriser les masques fiables
29  │ Intelligent MAX_MASKS                 │ ⬚ À FAIRE │ Éjecter les moins confiants
════╪═══════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 5 — STABILITÉ VISUELLE
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
30  │ Réduire clignotement                  │ ⬚ À FAIRE │ Combinaison smooth + prédiction
    │                                       │            │ + moins de faux positifs
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
31  │ Transition douce blur                 │ ⬚ À FAIRE │ Fade-in/out basé sur TTL
════╪═══════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 6 — STREAM LIVE
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
32  │ Test stabilité longue durée           │ ⬚ À FAIRE │ 30min+, RAM, FPS, CPU
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
33  │ Gestion transitions de jeu            │ ⬚ À FAIRE │ Replay, goal, menu, scoreboard
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
34  │ Premier stream live réel              │ ⬚ À FAIRE │ 🎯 Objectif final
════╪═══════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 7 — GPU (SI NÉCESSAIRE)
────┼───────────────────────────────────────┼────────────┼──────────────────────────────────
35  │ Pipeline GPU (CUDA / OpenCL)          │ ⬚ À FAIRE │ Si CPU insuffisant en live
════╪═══════════════════════════════════════╪════════════╪══════════════════════════════════

```

---

## Planning par sessions

```txt
══════════════════════════════════════════════════════════════════════════════════════════
                              PLANNING PAR SESSIONS
══════════════════════════════════════════════════════════════════════════════════════════

SESSION 1 — Config globale YAML (#22-23)               ✅ TERMINÉE (~30 min)
──────────────────────────────────────────────────────────────────────────────
  ✅ Créé config.yaml avec TOUS les paramètres
  ✅ Créé config.py singleton (from config import cfg)
  ✅ Migré main.py, detect.py, blur.py, capture/detect/send_thread.py
  ✅ Validé en PROD : ~49 FPS, zéro magic number

  Résultat : zéro constante hardcodée dans le code ✅

──────────────────────────────────────────────────────────────────────────────
SESSION 2 — Dual Detect + Fast Tracking (#10, 24, 27)  ✅ TERMINÉE
──────────────────────────────────────────────────────────────────────────────
  ✅ FastTrackThread : ROI redetect autour des masques connus
  ✅ Orchestration dual-thread (slow full + fast ROI)
  ✅ Hot-reload config (watcher thread)
  ✅ detect_stats thread-safe
  ✅ White masking (core + ext + dilate)
  ✅ Prédiction par vélocité
  ✅ Multi-mode blur (pixelate, box, gaussian, fill)

  Résultat : pipeline v8 opérationnel ✅

──────────────────────────────────────────────────────────────────────────────
SESSION 3 — Qualité détection (#28-29)                 ⬚ À PLANIFIER
──────────────────────────────────────────────────────────────────────────────
  • Score de confiance par masque
  • Éjection intelligente des masques les moins fiables

──────────────────────────────────────────────────────────────────────────────
SESSION 4 — Stabilité visuelle (#30-31)                ⬚ À PLANIFIER
──────────────────────────────────────────────────────────────────────────────
  • Réduction du clignotement
  • Transition douce blur (fade-in/out)

──────────────────────────────────────────────────────────────────────────────
SESSION 5 — Stream live (#32-34)                       ⬚ À PLANIFIER
──────────────────────────────────────────────────────────────────────────────
  • Test longue durée (30min+)
  • Gestion transitions (replay, goal, menu)
  • Premier stream live réel 🎯

──────────────────────────────────────────────────────────────────────────────
SESSION 6 — GPU (si nécessaire) (#35)                  ⬚ CONDITIONNEL
──────────────────────────────────────────────────────────────────────────────
  • Pipeline GPU uniquement si CPU insuffisant en live

══════════════════════════════════════════════════════════════════════════════════════════

CHEMIN CRITIQUE :

  S1 ✅ ──→ S2 ✅ ──→ S3 ──→ S4 ──→ S5 ──→ [S6 si besoin]
                      45min  1h    2h     2h

  Total restant chemin critique : ~5h45
  Total avec GPU                : ~7h45

══════════════════════════════════════════════════════════════════════════════════════════
```

---

## Dépendances

```bash
pip install opencv-python numpy pyvirtualcam dxcam pyyaml
```

| Lib             | Usage                            |
| --------------- | -------------------------------- |
| `opencv-python` | Traitement image (HSV, blur)     |
| `numpy`         | Buffers, opérations vectorielles |
| `pyvirtualcam`  | Envoi vers OBS Virtual Camera    |
| `dxcam`         | Capture écran DirectX (Windows)  |
| `pyyaml`        | Chargement config.yaml           |

---

## Lancement

```bash
python main.py
```

Ctrl+C pour arrêter proprement — le benchmark complet s'affiche à la sortie.
