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
CaptureThread ──→ frame BGR
    │
    ├──→ DetectThread (HSV dual-pass orange/bleu)
    │        └──→ zones [(x, y, w, h), ...]
    │
    ├──→ Main Loop (TTL + matching distance + smooth + prédiction)
    │        └──→ blur_zones
    │
    ├──→ blur.py (Pixelate in-place)
    │
    └──→ SendThread ──→ OBS Virtual Camera (pyvirtualcam)
```

---

## Fichiers

| Fichier             | Rôle                                              |
| ------------------- | ------------------------------------------------- |
| `main.py`           | Orchestration pipeline, TTL, matching, debug      |
| `config.py`         | Singleton config — charge config.yaml             |
| `config.yaml`       | Paramètres centralisés (zéro magic number)        |
| `capture_thread.py` | Thread de capture DXCam non-bloquant              |
| `detect_thread.py`  | Thread de détection HSV non-bloquant              |
| `detect.py`         | Pipeline HSV dual-pass (v2) + pipeline Sobel (v1) |
| `blur.py`           | Pixelate des zones + conversion BGR→RGB           |
| `send_thread.py`    | Thread d'envoi vers la caméra virtuelle OBS       |

---

## Paramètres principaux (`config.yaml`)

| Paramètre       | Valeur par défaut | Description                                        |
| --------------- | ----------------- | -------------------------------------------------- |
| `SCREEN_WIDTH`  | 1920              | Résolution capture                                 |
| `SCREEN_HEIGHT` | 1080              | Résolution capture                                 |
| `CAPTURE_FPS`   | 120               | FPS cible DXCam                                    |
| `VCAM_FPS`      | 120               | FPS déclaré à OBS                                  |
| `TTL_MAX`       | 4                 | Persistance d'un masque (en frames)                |
| `MARGIN`        | 0                 | Marge (px) autour de chaque zone floutée           |
| `MAX_MASKS`     | 10                | Nombre max de masques actifs simultanément         |
| `MATCHING_MODE` | `"distance"`      | Algo de matching : `"distance"` ou `"iou"`         |
| `DIST_THRESH`   | 60                | Seuil distance (px) pour matcher un masque         |
| `IOU_THRESH`    | 0.15              | Seuil IoU pour matcher un masque                   |
| `SMOOTH_FACTOR` | 0.4               | Lissage exponentiel des rects (0=fixe, 1=immédiat) |
| `DEBUG_DRAW`    | `false`           | Affiche les rectangles colorés TTL dans OBS        |
| `BLUR_MODE`     | `"pixelate"`      | Mode de flou : `"pixelate"` ou `"gaussian"`        |

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
 5  │ Morphologie (open + close)       │ v2.0    │ Nettoyage contours
 6  │ Filtres de forme (ratio, area)   │ v2.0    │ Élimine gros blocs
 7  │ DetectThread (non-bloquant)      │ v5.0    │ Detect découplé
 8  │ Resize avant detect (SCALE)      │ v5.0    │ Accélère le HSV
────┼──────────────────────────────────┼─────────┼────────────────────────
                    SUIVI / MASQUES
────┼──────────────────────────────────┼─────────┼────────────────────────
 9  │ Système TTL (persistance)        │ v4.0    │ Masque survit N frames
10  │ Matching par distance            │ v4.0    │ Réidentifie un masque
11  │ Matching par IoU                 │ v4.0    │ Alternative au distance
12  │ MAX_MASKS (cap masques actifs)   │ v5.0    │ Évite explosion
13  │ Vélocité par masque              │ v6.0    │ dx/dy entre détections
14  │ Prédiction linéaire              │ v6.0    │ Extrapole entre detects
15  │ Smooth factor (lissage rects)    │ v6.0    │ Élimine micro-sauts
16  │ TTL_MAX réduit à 4               │ v6.0    │ Moins de fantômes
────┼──────────────────────────────────┼─────────┼────────────────────────
                    RENDU / BLUR
────┼──────────────────────────────────┼─────────┼────────────────────────
17  │ GaussianBlur                     │ v1.0    │ Premier mode de flou
18  │ Pixelate (mode actuel)           │ v5.0    │ Plus rapide, plus net
19  │ Blur in-place                    │ v3.0    │ Pas de copie frame
20  │ Marge configurable (MARGIN)      │ v4.0    │ Agrandit la zone blur
────┼──────────────────────────────────┼─────────┼────────────────────────
                    SORTIE
────┼──────────────────────────────────┼─────────┼────────────────────────
21  │ Envoi pyvirtualcam → OBS         │ v1.0    │ Caméra virtuelle
22  │ SendThread (non-bloquant)        │ v5.0    │ Envoi découplé
23  │ Double buffer SendThread         │ v6.0    │ Zéro tearing
24  │ RGB buffer pré-alloué            │ v5.0    │ Évite malloc/frame
────┼──────────────────────────────────┼─────────┼────────────────────────
                    CONFIG / INFRA
────┼──────────────────────────────────┼─────────┼────────────────────────
25  │ config.yaml centralisé           │ v6.1    │ Zéro magic number
26  │ config.py singleton              │ v6.1    │ Import unique partout
27  │ DEBUG_DRAW (rects colorés TTL)   │ v5.0    │ Visualise les masques
28  │ Benchmark intégré (tous modules) │ v5.0    │ Stats à la sortie
29  │ Skip frames (configurable)       │ v5.0    │ Blur 1/N frames
══════════════════════════════════════════════════════════════════════════
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
 #  │ Action                              │ Statut     │ Notes
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                         PHASE 1 — FONDATIONS (✅ TERMINÉE)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 1  │ Capture écran (DXCam)               │ ✅ FAIT    │ 120fps, grab_avg ~28ms
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 2  │ Détection couleur HSV               │ ✅ FAIT    │ Orange + Bleu, morpho, contours
    │ (cartouches noms)                   │            │ detect_avg ~102ms
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 3  │ Blur des zones détectées            │ ✅ FAIT    │ Pixelate in-place
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 4  │ Envoi vers caméra virtuelle OBS     │ ✅ FAIT    │ pyvirtualcam + OBS Virtual Camera
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 5  │ Pipeline fonctionnel bout-en-bout   │ ✅ FAIT    │ capture → detect → blur → vcam
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                     PHASE 2 — OPTIMISATION PIPELINE (✅ TERMINÉE)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 6  │ Thread séparé pour la détection     │ ✅ FAIT    │ DetectThread, non-bloquant
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 7  │ Thread séparé pour l'envoi vcam     │ ✅ FAIT    │ SendThread, non-bloquant
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 8  │ Système TTL (persistance masques)   │ ✅ FAIT    │ TTL_MAX = 4
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
 9  │ Matching distance (suivi masques)   │ ✅ FAIT    │ DIST_THRESH = 60
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
10  │ Skip frames (blur 1/N)              │ ✅ FAIT    │ SKIP = 1 (désactivé)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
11  │ Resize avant détection (SCALE)      │ ✅ FAIT    │ SCALE = 2.0, remap coords
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
12  │ Benchmark intégré (tous modules)    │ ✅ FAIT    │ Stats par module + récap final
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
13  │ Mode DEBUG_DRAW                     │ ✅ FAIT    │ Rects colorés TTL + labels
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
14  │ RGB buffer pré-alloué               │ ✅ FAIT    │ Évite allocation chaque frame
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                PHASE 2.5 — INTÉGRITÉ MÉMOIRE (✅ TERMINÉE)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
15  │ Double buffer SendThread            │ ✅ FAIT    │ Copie interne, zéro tearing
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
16  │ Valider : pas de tearing/artefacts  │ ✅ FAIT    │ Test visuel PROD OK
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                PHASE 2.7 — SUIVI & PRÉDICTION MASQUES (✅ TERMINÉE)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
17  │ SMOOTH_FACTOR sur positions         │ ✅ FAIT    │ α = 0.4, élimine micro-sauts
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
18  │ Vélocité par masque                 │ ✅ FAIT    │ dx/dy calculé à chaque match
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
19  │ Prédiction linéaire                 │ ✅ FAIT    │ pos = pos + v × Δframes
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                PHASE 2.9 — CONFIG GLOBALE (✅ TERMINÉE)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
20  │ Fichier config.yaml                 │ ✅ FAIT    │ Tous les paramètres centralisés
    │                                     │            │ Chargé au démarrage par config.py
    │                                     │            │ Partagé avec tous les modules
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
21  │ config.py singleton                 │ ✅ FAIT    │ from config import CFG partout
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
22  │ Valider pipeline complet            │ ✅ FAIT    │ ~49 FPS PROD, zéro magic number
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                PHASE 3 — DOUBLE RÉSOLUTION
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
23  │ Detect rapide (SCALE=4)             │ 🟡 NEXT   │ ~15-20ms, grossier
    │                                     │            │ Confirme les masques existants
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
24  │ Detect lent (SCALE=2, actuel)       │ ⬚ À FAIRE │ ~100ms, précis
    │                                     │            │ Découvre + recale les masques
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
25  │ Orchestration dual-detect           │ ⬚ À FAIRE │ Thread rapide : continu (positions)
    │                                     │            │ Thread lent : 1/N (découverte)
    │                                     │            │ Main fusionne les deux résultats
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
26  │ Valider dual-detect                 │ ⬚ APRÈS   │ Detect effectif >25Hz
    │                                     │ #25        │ Qualité maintenue
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                  PHASE 4 — RÉDUCTION FAUX POSITIFS
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
27  │ Diagnostic faux positifs (diag.py)  │ ⬚ À FAIRE │ Replay, goal, menu, scoreboard
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
28  │ Resserrer seuils HSV               │ ⬚ À FAIRE │ Basé sur mesures diag
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
29  │ Ajuster filtres de forme            │ ⬚ À FAIRE │ Dimensions, ratio, fill
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
30  │ Cap intelligent MAX_MASKS           │ ⬚ À FAIRE │ Score de confiance
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 5 — STABILITÉ VISUELLE
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
31  │ Réduire clignotement                │ ⬚ À FAIRE │ Combinaison smooth + prédiction
    │                                     │            │ + moins de faux positifs
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
32  │ Transition douce blur               │ ⬚ À FAIRE │ Fade-in/out basé sur TTL
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 6 — STREAM LIVE
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
33  │ Test stabilité longue durée         │ ⬚ À FAIRE │ 30min+, RAM, FPS, CPU
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
34  │ Gestion transitions de jeu          │ ⬚ À FAIRE │ Replay, goal, menu, scoreboard
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
35  │ Premier stream live réel            │ ⬚ À FAIRE │ 🎯 Objectif final
════╪═════════════════════════════════════╪════════════╪══════════════════════════════════
                    PHASE 7 — GPU (SI NÉCESSAIRE)
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
36  │ Évaluer le besoin GPU               │ ⬚ CONDITI  │ Seulement si <30 FPS après
    │                                     │ ONNEL      │ phases 3 + 4 + 5
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
37  │ Pipeline HSV sur CUDA               │ ⬚ CONDITI  │ cv2.cuda : cvtColor, inRange,
    │                                     │ ONNEL      │ morphologyEx → ~5-10ms
────┼─────────────────────────────────────┼────────────┼──────────────────────────────────
38  │ Blur sur CUDA                       │ ⬚ CONDITI  │ cv2.cuda.GaussianBlur → ~1ms
    │                                     │ ONNEL      │
══════════════════════════════════════════════════════════════════════════════════════════
```

---

## Planning par sessions

```txt
══════════════════════════════════════════════════════════════════════════════════════════
                              PLANNING PAR SESSIONS
══════════════════════════════════════════════════════════════════════════════════════════

SESSION 1 — Config globale YAML (#20-22)               ✅ TERMINÉE (~30 min)
──────────────────────────────────────────────────────────────────────────────
  ✅ Créé config.yaml avec TOUS les paramètres
  ✅ Créé config.py singleton (from config import CFG)
  ✅ Migré main.py, detect.py, blur.py, capture/detect/send_thread.py
  ✅ Validé en PROD : ~49 FPS, zéro magic number

  Résultat : zéro constante hardcodée dans le code ✅

──────────────────────────────────────────────────────────────────────────────
SESSION 2 — Double Résolution (#23-26)                 Durée estimée : 1h30
──────────────────────────────────────────────────────────────────────────────
  Étape A — detect_fast() standalone (30 min)
  • SCALE=4, seuils relâchés → viser <20ms
  • Vérifier qu'il retrouve les masques déjà connus

  Étape B — Orchestration dual-thread (45 min)
  • DetectFastThread : continu, met à jour positions
  • DetectSlowThread : 1/3 frames, découvre + recale
  • Main fusionne les deux flux

  Étape C — Validation (15 min)
  • Detect effectif passe de ~10Hz à ~30Hz
  • Pas de régression qualité

  Critère de fin : detect effectif >25Hz, qualité maintenue
  Dépend de : session 1 ✅

──────────────────────────────────────────────────────────────────────────────
SESSION 3 — Diagnostic faux positifs (#27)             Durée estimée : 45 min
──────────────────────────────────────────────────────────────────────────────
  • Lancer sur : en jeu, replay, scoreboard, goal, menu
  • Capturer screenshots annotés
  • Rédiger faux_positifs.md

  Critère de fin : document de référence rédigé
  ⚡ Parallélisable avec session 2

──────────────────────────────────────────────────────────────────────────────
SESSION 4 — Nettoyage détection (#28-31)               Durée estimée : 1h
──────────────────────────────────────────────────────────────────────────────
  • Exclusion zones écran (via config.yaml)
  • Ajuster seuils HSV + filtres forme (via config.yaml)
  • Cap intelligent par score de confiance
  • Valider en DEBUG_DRAW sur toutes les situations

  Critère de fin : <2 faux positifs en jeu normal
  Dépend de : sessions 1 ✅ + 3

──────────────────────────────────────────────────────────────────────────────
SESSION 5 — Stabilité visuelle (#32-33)                Durée estimée : 45 min
──────────────────────────────────────────────────────────────────────────────
  • Mesurer clignotement restant
  • Implémenter fade-in/fade-out basé sur TTL
  • Test visuel 10 min continu

  Critère de fin : flux propre pour un spectateur
  Dépend de : sessions 2 + 4

──────────────────────────────────────────────────────────────────────────────
SESSION 6 — Pré-production (#34-35)                    Durée estimée : 1h
──────────────────────────────────────────────────────────────────────────────
  • Test stabilité 30 min (RAM, FPS, CPU)
  • Tester transitions : replay, goal, menu
  • Ajustements config.yaml finaux

  Critère de fin : 30 min sans problème

──────────────────────────────────────────────────────────────────────────────
SESSION 7 — Premier live (#36)                         Durée estimée : 2h
──────────────────────────────────────────────────────────────────────────────
  • Stream test privé 30 min → relecture VOD
  • Ajustements finaux
  • Premier vrai stream 🎯

  Critère de fin : VOD regardable, noms invisibles

──────────────────────────────────────────────────────────────────────────────
SESSION 8 — GPU (CONDITIONNELLE) (#37-39)              Durée estimée : 2h
──────────────────────────────────────────────────────────────────────────────
  ⚠️  SEULEMENT SI après sessions 1-6 :
      - FPS < 30 en PROD
      - OU latence perçue encore gênante
      - OU CPU > 80% en continu

  • Installer opencv-contrib avec CUDA
  • Porter pipeline HSV sur cv2.cuda
  • Porter blur sur cv2.cuda
  • Benchmark comparatif CPU vs GPU

  Critère de fin : gain mesuré justifie la complexité ajoutée

══════════════════════════════════════════════════════════════════════════════════════════

CHEMIN CRITIQUE :

  S1 ✅ ──→ S2 ──→ S5 ──→ S6 ──→ S7 ──→ [S8 si besoin]
  30min ✅  1h30   45min  1h     2h       2h
    │
    └──→ S3 ──→ S4 ──────┘
         45min  1h
         (parallélisable avec S2)

  Total chemin critique : ~6h (restant après S1)
  Total avec GPU        : ~8h

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
