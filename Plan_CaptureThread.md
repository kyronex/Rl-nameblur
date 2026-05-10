# Plan — Capture hybride DXCam → cv2 fallback

---

## Phase 0 — Audit `capture_thread.py`

**Objectif** : cartographier l'existant avant toute modification.

Points à relever :

- Localisation de l'initialisation DXCam (`__init__` vs `run()`)
- Localisation du `get_latest_frame()` et de la boucle principale
- API publique exposée : `get_frame()`, `get_frame_id()`, `_latest_frame`, `_latest_ts`, `_frame_id`
- Dépendances entrantes depuis `main.py` et autres threads
- Présence ou absence d'un mécanisme de stop propre (`threading.Event` ou équivalent)
- Format de sortie actuel (RGB numpy array, shape, dtype)

**Livrable** : liste des points d'insertion identifiés + confirmation que l'API publique est stable.

---

### Phase 1 — Abstraction de la source capture

**Objectif** : isoler la logique de capture derrière une interface commune pour que `CaptureThread` soit agnostique de la source.

Structure cible :

```text
capture/
    __init__.py
    base.py          ← classe abstraite CaptureSource
    dxcam_source.py  ← wrapper DXCam
    cv2_source.py    ← wrapper cv2.VideoCapture
```

Interface `CaptureSource` (base.py) :

- `start() → None`
- `grab() → np.ndarray | None` — retourne frame RGB ou None
- `stop() → None`
- `@property fps_target → int`
- `@property resolution → tuple[int, int]`

**Contrainte** : `CaptureThread` ne connaît que `CaptureSource`. Zéro import DXCam ou cv2 dans `CaptureThread`.

---

### Phase 2 — Implémentation `DXCamSource`

Points couverts :

- Wrapping de l'init DXCam (device_idx, output_idx, region depuis config)
- `grab()` retourne frame RGB numpy ou None
- Gestion propre `start()` / `stop()`
- Log du device sélectionné au démarrage (aide debug setup streamer)

---

### Phase 3 — Implémentation `Cv2Source`

Points couverts :

**Détection de l'index OBS Virtual Camera** :

- Utiliser `comtypes` (déjà dépendance) pour lister les périphériques DirectShow
- Filtrer sur le nom contenant `"OBS"` (case-insensitive)
- Fallback : scan index 0..9 si listing DirectShow échoue
- Log du nom et de l'index retenu

**Validation à l'ouverture** :

- Résolution retournée par cv2 == `config.screen.width × config.screen.height`
- WARNING explicite si mismatch (streamer a mal configuré OBS)

**Boucle de lecture** :

- `read()` bloquant dans le worker
- Timeout sur `read()` pour ne pas bloquer indéfiniment si OBS coupe le flux

---

### Phase 4 — Logique de sélection automatique (`SourceSelector`)

**Objectif** : encapsuler la logique de détection et fallback dans un composant dédié, pas dans `CaptureThread`.

Séquence :

```text
SourceSelector.resolve(config) → CaptureSource

    1. Tentative DXCamSource
           └── start()
           └── Probe : N tentatives grab() sur T secondes (configurable)
                   ├── Au moins 1 frame valide → DXCamSource retourné ✅
                   └── 0 frame valide → DXCamSource.stop()
                               ↓
    2. Tentative Cv2Source
           └── Détection index OBS (comtypes → scan)
           └── Validation résolution
                   ├── Valide → Cv2Source retourné ✅
                   └── Invalide → Cv2Source.stop()
                               ↓
    3. Échec total
           └── Log ERROR actionnable (mode fenêtre, OBS non lancé)
           └── raise CaptureSourceNotFound  ← exit propre dans main.py
```

Paramètres configurables dans `config.yaml` (section `capture`) :

- `probe_timeout_s` : durée max de la probe DXCam (défaut : 0.5s)
- `probe_min_frames` : frames valides min pour valider DXCam (défaut : 1)
- `source_priority` : `["dxcam", "cv2"]` — ordre de tentative, modifiable

**Évolutivité** : ajouter WGC plus tard = ajouter `wgc_source.py` + entrée dans `source_priority`.

---

### Phase 5 — Intégration dans `CaptureThread`

Modifications dans `capture_thread.py` :

- `__init__` appelle `SourceSelector.resolve(config)` → reçoit un `CaptureSource`
- La boucle principale appelle `self._source.grab()`
- `stop()` appelle `self._source.stop()`
- **Zéro autre changement** — `get_frame()`, `get_frame_id()`, `_latest_frame`, `_latest_ts`, `_frame_id` inchangés

---

### Phase 6 — Gestion d'erreur et messages utilisateur

**Exit immédiat (V1)** :

- `CaptureSourceNotFound` remontée jusqu'à `main.py`
- Message utilisateur structuré :

  ```text
  [ERREUR] Aucune source de capture disponible.

  Solutions :
  1. Lancez Rocket League en mode Borderless Window
  2. OU lancez OBS avec une source "Game Capture" et activez la Virtual Camera

  Puis relancez le script.
  ```

- Exit code non-zéro

**Évolution V2 (tracée, pas implémentée)** :

- Retry loop avec intervalle dans `main.py`
- `SourceSelector.resolve()` devient `SourceSelector.wait_for_source(timeout_s)`

---

### Phase 7 — Config `config.yaml`

Nouvelle section à ajouter :

```yaml
capture:
  source_priority: ["dxcam", "cv2"]
  probe_timeout_s: 0.5
  probe_min_frames: 1
  obs_virtual_camera_name: "OBS" # substring recherchée dans le nom DirectShow
```

Invariants à ajouter dans la validation config (cohérent avec A-02 du Plan.md) :

- `probe_timeout_s > 0`
- `probe_min_frames >= 1`
- `source_priority` non vide, valeurs dans `["dxcam", "cv2"]`

---

### Phase 8 — Documentation

- `README.md` : mise à jour section **Dépendances** (aucun ajout, `comtypes` déjà présent)
- `README.md` : mise à jour section **Lancement** — prérequis source (Borderless ou OBS Virtual Camera)
- `config.yaml` : commentaires inline sur les nouvelles clés
- Note explicite : **bascule DXCam → FSE post-démarrage non gérée en V1**, relance script requise

---

### Séquence d'exécution

| Phase                         | Dépendance   | Durée estimée |
| ----------------------------- | ------------ | ------------- |
| 0 — Audit                     | Aucune       | 30 min        |
| 1 — Abstraction base          | Phase 0      | 30 min        |
| 2 — DXCamSource               | Phase 1      | 30 min        |
| 3 — Cv2Source                 | Phase 1      | 1h            |
| 4 — SourceSelector            | Phases 2 + 3 | 1h            |
| 5 — Intégration CaptureThread | Phase 4      | 30 min        |
| 6 — Gestion erreur            | Phase 5      | 30 min        |
| 7 — Config                    | Phase 4      | 15 min        |
| 8 — Doc                       | Phase 5      | 30 min        |

**Total estimé :** 5h

---

### Ce que ce plan ne change pas

- API publique `CaptureThread` : inchangée
- Tous les threads aval (`DetectThread`, `FastTrackThread`, `SendThread`) : inchangés
- `main.py` : un seul ajout — catch `CaptureSourceNotFound` → exit propre
- `config.yaml` : section additive, aucune clé existante modifiée
