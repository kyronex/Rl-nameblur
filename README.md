# RL-NameBlur

Script Python pour anonymiser en temps réel les noms des joueurs dans Rocket League, via une caméra virtuelle (OBS).

---

## Fonctionnement général

```text
Écran (Rocket League)
        │
        ▼
┌───────────────┐
│ CaptureThread │  dxcam — capture GPU @ 120fps
└───────┬───────┘
        │ frame BGR
        ├──────────────────────────────┐
        ▼                              ▼
┌───────────────┐              ┌───────────────┐
│  DetectThread │              │   Main Loop   │
│  detect.py    │              │   main.py     │
│  HSV dual-pass│◄─────────────│ give_frame()  │
│  ~15 FPS      │              │               │
└───────┬───────┘              └───────┬───────┘
        │ plates [(x,y,w,h)]           │
        └──────────────────────────────┤
                                       │ TTL + IoU matching
                                       │ active_masks
                                       ▼
                              ┌───────────────┐
                              │   blur.py     │
                              │ apply_blur()  │
                              │ BGR → RGB     │
                              └───────┬───────┘
                                      │ frame RGB
                                      ▼
                              ┌───────────────┐
                              │  SendThread   │
                              │  vcam.send()  │
                              │  → OBS        │
                              └───────────────┘
```

---

## Architecture des fichiers

```text
rl-nameblur/
├── main.py              # Boucle principale, orchestration des threads, TTL + IoU
├── capture_thread.py    # Thread de capture écran via dxcam (non bloquant)
├── detect_thread.py     # Thread de détection HSV (non bloquant)
├── detect.py            # Pipeline de détection (V1 Sobel + V2 HSV dual-pass)
├── blur.py              # Application du flou gaussien + conversion BGR→RGB
├── send_thread.py       # Thread d'envoi vers la caméra virtuelle OBS
└── README.md
```

---

## Pipeline de détection — detect.py

### V2 HSV dual-pass (pipeline actif)

```text
Frame BGR
    │
    ▼
Resize ÷2 (SCALE=2.0)          → gain CPU ~75%
    │
    ▼
Conversion BGR → HSV
    │
    ├──► Masque Orange  [H:8-22  S:140-255 V:170-255]
    ├──► Masque Bleu    [H:100-125 S:130-255 V:150-255]
    └──► Masque Blanc   [H:0-180  S:0-60   V:200-255]
    │
    ▼
Morphologie (fermeture H + V)   → combler les trous dans les cartouches
    │
    ▼
Fusion Orange | Bleu
    │
    ▼
AND avec Blanc dilaté           → garder uniquement les zones avec du texte
    │
    ▼
findContours
    │
    ▼
Filtre forme :
  - Aire min/max
  - Largeur / Hauteur min/max
  - Ratio w/h : [2.0 – 15.0]   → cartouche = rectangle horizontal
  - Fill ratio > 0.35
    │
    ▼
Remap × SCALE → coordonnées originales
    │
    ▼
plates [(x, y, w, h), ...]
```

### V1 Sobel (pipeline legacy, conservé pour comparaison)

```text
Frame BGR → Resize → Grayscale → GaussianBlur
    → Sobel Y → Threshold → Dilate → Contours
    → Filtre forme → Filtre enfants → Validation HSV → plates
```

---

## Gestion des masques — main.py

### TTL (Time To Live)

Chaque zone détectée reçoit un compteur de vie. Si la détection disparaît
(mouvement rapide, faux négatif), le masque reste actif quelques frames.

```python
TTL_MAX = 8         → durée de vie initiale
TTL décrément       → -1 à chaque nouvelle détection reçue
TTL = 0             → masque supprimé
```

### IoU matching (en cours de remplacement par distance de centre)

Quand une nouvelle détection arrive, on cherche si elle correspond à un
masque existant via l'Intersection over Union.

```text
IoU ≥ IOU_THRESH (0.15) → mise à jour du masque existant (TTL reset)
IoU < IOU_THRESH        → nouveau masque créé
```

**Limitation connue** : en mouvement rapide, le décalage du rect fait chuter
l'IoU sous le seuil → doublon créé → l'ancien masque meurt → scintillement.
→ Remplacement prévu par matching via distance de centre.

### Cycle complet par frame

```text
1. get_frame()          → dernière frame capturée (non bloquant)
2. give_frame()         → envoi au DetectThread
3. get_detect_count()   → vérifier si nouvelle détection disponible
4. match_or_add()       → IoU matching + TTL reset ou nouveau masque
5. TTL décrément        → vieillissement de tous les masques actifs
6. Purge TTL=0          → suppression des masques morts
7. Cap MAX_MASKS=20     → limite de sécurité
8. apply_blur()         → flou sur les zones actives + BGR→RGB
9. give_frame()         → envoi au SendThread → OBS
```

---

## Paramètres configurables

### Capture

| Paramètre     | Valeur | Description                       |
| ------------- | ------ | --------------------------------- |
| SCREEN_WIDTH  | 1920   | Largeur de l'écran capturé        |
| SCREEN_HEIGHT | 1080   | Hauteur de l'écran capturé        |
| CAPTURE_FPS   | 120    | FPS cible de la capture dxcam     |
| VCAM_FPS      | 120    | FPS déclaré à la caméra virtuelle |

### Détection — detect.py

| Paramètre | Valeur | Description                            |
| --------- | ------ | -------------------------------------- |
| SCALE     | 2.0    | Facteur de réduction avant traitement  |
| MIN_RATIO | 2.0    | Ratio w/h minimum d'une cartouche      |
| MAX_RATIO | 15.0   | Ratio w/h maximum d'une cartouche      |
| MIN_FILL  | 0.35   | Taux de remplissage minimum du contour |

### Masques — main.py

| Paramètre  | Valeur | Description                              |
| ---------- | ------ | ---------------------------------------- |
| TTL_MAX    | 8      | Durée de vie d'un masque (en détections) |
| MARGIN     | 6      | Padding en pixels autour du rect détecté |
| IOU_THRESH | 0.15   | Seuil de matching IoU                    |
| MAX_MASKS  | 20     | Nombre maximum de masques simultanés     |
| SKIP       | 1      | Appliquer le blur 1 frame sur N          |

### Flou — blur.py

| Paramètre     | Valeur | Description                        |
| ------------- | ------ | ---------------------------------- |
| BLUR_STRENGTH | 51     | Taille du kernel gaussien (impair) |
| MARGIN        | -2     | Ajustement fin du rect avant flou  |

---

## Mode DEBUG

Activé via `DEBUG_DRAW = True` dans `main.py`.

Affiche les rectangles de détection sans appliquer le flou.
Utile pour valider la détection avant de passer en production.

```text
🟩 Vert   TTL ≥ 3   détection fraîche et stable
🟨 Jaune  TTL = 2   masque en train de vieillir
🟥 Rouge  TTL = 1   masque mourant (sera supprimé à la prochaine détection)
```

---

## Performances mesurées

| Étape         | Temps moyen | Notes                       |
| ------------- | ----------- | --------------------------- |
| Capture       | ~1-2 ms     | dxcam GPU, thread dédié     |
| Détection V2  | ~15-20 ms   | HSV dual-pass, SCALE÷2      |
| Blur + CVT    | ~2-4 ms     | GaussianBlur + BGR→RGB      |
| Send vcam     | ~1-2 ms     | Thread dédié                |
| Main loop     | ~5-8 ms     | Orchestration uniquement    |
| **FPS total** | **~35-50**  | Dépend du nombre de masques |

---

## Dépendances

```python
dxcam          # Capture écran GPU (Windows uniquement)
opencv-python  # Traitement image (HSV, morpho, blur, contours)
pyvirtualcam   # Caméra virtuelle → OBS
numpy          # Buffers image
```

---

## Limitations connues et évolutions prévues

| #   | Limitation                                      | Statut             |
| --- | ----------------------------------------------- | ------------------ |
| 1   | Scintillement en mouvement rapide (IoU fragile) | 🔧 En cours        |
| 2   | Faux positifs sur éléments orange/bleu du HUD   | ⚠️ À filtrer       |
| 3   | Pas de filtre par zone de l'écran               | ⚠️ À ajouter       |
| 4   | Résolution fixe 1920×1080                       | ℹ️ Non prioritaire |
| 5   | Windows uniquement (dxcam)                      | ℹ️ Par conception  |

---

## Prochaine évolution : matching par distance de centre

Remplacement du matching IoU par distance euclidienne entre centres des rects.

```python
# Avant
IoU >= 0.15  → match

# Après
distance(centre1, centre2) <= 80px  → match
```

Avantage : robuste aux décalages de position en mouvement rapide,
sans risque de fusionner deux noms distincts proches l'un de l'autre.
