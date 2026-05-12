# Plan — Capture hybride DXCam → cv2 fallback

> **Document autoportant.** Aucune référence implicite aux plans antérieurs. En cas de doute sur le périmètre exact d'un item, un **audit de code base** est explicitement requis avant démarrage.

---

## Décisions architecturales figées

| Décision                         | Choix V1                                                | Justification                                                                            |
| -------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Contrat `CaptureSource`          | **A1 strict** (4 membres)                               | Robustesse maximale via surface minimale. Un seul comportement à implémenter par source. |
| Stratégie probe                  | **B1 full lifecycle** (`start()` + `grab()` + `stop()`) | Un seul chemin de code testé. Le seul vrai test de disponibilité = obtenir une frame.    |
| État source post-`resolve()`     | **β — source arrêtée**                                  | Séparation responsabilités `SourceSelector` ↔ `CaptureThread`. Pas d'état caché. Coût:   |
|                                  |                                                         | Coût: +300 ms démarrage (acceptable, une fois).                                          |
| Idempotence `start()` / `stop()` | **Requise** (Phases 2 et 3)                             | Conséquence directe de β — appelés deux fois (probe + run).                              |
| Conversion couleur               | **Dans la source**                                      | Format de sortie RGB obligatoire au contrat. `Cv2Source` fait BGR→RGB.                   |
| Injection `CaptureSource`        | **Option (a) — injection externe**                      | `main.py` call `SourceSelector.resolve()` puis instancie `CaptureThread(source=...)`.    |
|                                  |                                                         | `CaptureSourceNotFound` levée **avant** instanciation du thread.                         |
|                                  |                                                         | Thread = consommateur passif.                                                            |

---

## Phase 0 — Audit `capture_thread.py` ✅ LIVRABLE

**Statut** : ✅ Phase 0 livrable. Aucun obstacle identifié à l'enchaînement Phase 1.

**Constats validés** :

- **6 sites de couplage DXCam** identifiés, tous concentrés dans `capture_thread.py` :
  1. `import dxcam` (ligne 3)
  2. `dxcam.device_info()` (ligne 16)
  3. `dxcam.create(output_color="RGB")` (ligne 18)
  4. `self._camera.start(target_fps=...)` (ligne 19)
  5. `self._camera.stop()` (ligne 32)
  6. `self._camera.get_latest_frame()` (ligne 49)

- **Init DXCam** : dans `start()` (pas `__init__`), avant démarrage du worker thread.
- **Stop signal** : flag bool `self._running` (pas `threading.Event`) — acceptable V1, worker poll à 1 ms.
- **Format sortie** : `np.ndarray`, `(H, W, 3)`, `dtype=uint8`, **RGB** (forcé par `output_color="RGB"`), contigu via `.copy()`. ✅ Conforme contrat A1.
- **API publique stable** : `start()`, `stop()`, `get_frame()`, `get_frame_id()` — signatures préservées par le refactor.
- **Diagnostic FPS réel** : log toutes les 5 s dans `_worker()` — à généraliser via `self._source.name`.

**Points de vigilance tracés** :

- ⚠️ **Grep requis avant Phase 5** sur `_latest_frame|_latest_ts|_frame_id` pour confirmer accès uniquement via getters (pas d'accès direct externe). Si accès direct trouvé → API à figer plus largement.
- ⚠️ **Log `dxcam.device_info()` dupliqué** si `start()` appelé deux fois (probe + run). À gérer dans `DXCamSource` (log conditionnel ou flag « déjà loggué »).

**Décision tranchée** : injection `CaptureSource` externe (option a, cf. Décisions architecturales figées).

---

## Phase 1 — Abstraction de la source capture

**Préconditions** : Phase 0 livrée ✅.

**Objectif** : isoler la logique de capture derrière une interface commune pour que `CaptureThread` soit agnostique de la source.

Structure cible :

```text
capture/
    __init__.py
    base.py          ← classe abstraite CaptureSource (contrat A1 strict)
    dxcam_source.py  ← wrapper DXCam
    cv2_source.py    ← wrapper cv2.VideoCapture
    selector.py      ← SourceSelector (Phase 4)
```

**Contrat `CaptureSource` (A1 strict — 4 membres, non négociable)** :

- `start(target_fps: int) → None` — peut lever si la source est indisponible.
- `grab() → np.ndarray | None` — frame RGB `(H, W, 3)` `uint8`, ou `None` si pas de frame disponible. Non bloquant. Ne lève pas en régime nominal.
- `stop() → None` — idempotent.
- `@property name → str` — identifiant court pour les logs (ex. `"dxcam"`, `"cv2"`).

**Format de sortie obligatoire** : `np.ndarray` RGB, `(H, W, 3)`, `dtype=uint8`. Toute conversion BGR→RGB se fait **dans la source** (notamment `Cv2Source`).

**Contrainte d'étanchéité** : `CaptureThread` ne connaît que `CaptureSource`. Zéro import DXCam ou cv2 dans `CaptureThread`.

---

## Phase 2 — Implémentation `DXCamSource`

**Préconditions** : Phase 1 livrée.

Points couverts :

- Wrapping de l'init DXCam (device_idx, output_idx, region depuis config)
- `start(target_fps)` configure et démarre le stream DXCam
- `grab()` retourne frame RGB numpy ou None (DXCam délivre déjà du RGB, conforme audit Phase 0)
- `stop()` idempotent — gère le cas appelé deux fois (probe + start réel)
- Log `dxcam.device_info()` **une seule fois** (flag interne) — cf. point de vigilance Phase 0
- `name` retourne `"dxcam"`

---

## Phase 3 — Implémentation `Cv2Source`

**Préconditions** : Phase 1 livrée.

Points couverts :

**Détection de l'index OBS Virtual Camera** :

- Utiliser `comtypes` (déjà dépendance) pour lister les périphériques DirectShow
- Filtrer sur le nom contenant la substring configurée (`obs_virtual_camera_name`, case-insensitive)
- Fallback : scan index 0..9 si listing DirectShow échoue
- Log du nom et de l'index retenu

**Validation à l'ouverture** :

- Résolution retournée par cv2 == `config.screen.width × config.screen.height`
- WARNING explicite si mismatch (streamer a mal configuré OBS)

**Conversion couleur** :

- `cv2.VideoCapture.read()` retourne du BGR → conversion BGR→RGB obligatoire dans `grab()` avant retour.

**Boucle de lecture** :

- `read()` dans le worker
- Timeout sur `read()` pour ne pas bloquer indéfiniment si OBS coupe le flux

**Notes de robustesse** :

- `target_fps` passé à `start()` est **indicatif** côté cv2 — le débit réel est imposé par OBS Virtual Camera. À documenter dans le module.
- `stop()` idempotent — gère le cas appelé deux fois (probe + start réel).
- `name` retourne `"cv2"`.

---

## Phase 4 — Logique de sélection automatique (`SourceSelector`)

**Préconditions** : Phases 2 et 3 livrées.

**Objectif** : encapsuler la logique de détection et fallback dans un composant dédié, pas dans `CaptureThread`.

**Stratégie de probe : B1 full lifecycle** — un seul chemin de code, testé sur l'usage réel.

Séquence :

```text
SourceSelector.resolve(config) → CaptureSource (arrêtée, prête à start)

    Pour chaque source dans config.capture.source_priority :
        1. Instanciation de la source
        2. source.start(target_fps=config.screen.capture_fps)
              └── exception → log warning, continue boucle
        3. Probe par grab() : attendre N frames valides en T secondes max
              ├── ≥ probe_min_frames frames valides en ≤ probe_timeout_s
              │       → source.stop()                        ← option β
              │       → return source (arrêtée, validée) ✅
              └── timeout → log warning
                          → source.stop()
                          → continue boucle

    Aucune source validée :
        └── log ERROR actionnable (mode fenêtre, OBS non lancé)
        └── raise CaptureSourceNotFound  ← exit propre dans main.py
```

**Décision architecturale (Option β)** : la source retournée par `resolve()` est **arrêtée**. `CaptureThread` rappelle `source.start(target_fps)` avant la boucle de capture.

- **Conséquence** : un cycle start/stop supplémentaire (~300 ms) au démarrage.
- **Bénéfice** : séparation claire des responsabilités, aucun état caché entre `SourceSelector` et `CaptureThread`. Idempotence de `start()` / `stop()` requise dans chaque source (Phases 2 et 3).

Paramètres configurables dans `config.yaml` (section `capture`) :

- `probe_timeout_s` : durée max de la probe (défaut : 0.5 s)
- `probe_min_frames` : frames valides min pour valider la source (défaut : 1)
- `source_priority` : `["dxcam", "cv2"]` — ordre de tentative, modifiable
- `obs_virtual_camera_name` : substring recherchée dans le nom DirectShow

**Évolutivité** : ajouter WGC, NDI, Spout = ajouter `xxx_source.py` + entrée dans `source_priority`. Aucune modification du `SourceSelector` requise grâce à A1 strict.

---

## Phase 5 — Intégration dans `CaptureThread`

**Préconditions** : Phase 4 livrée + grep `_latest_frame|_latest_ts|_frame_id` effectué (point de vigilance Phase 0).

**Points d'insertion (validés par audit Phase 0)** :

| Modification                   | Localisation actuelle            | Localisation cible                                                     |
| ------------------------------ | -------------------------------- | ---------------------------------------------------------------------- |
| Suppression `import dxcam`     | Ligne 3                          | Remplacé par `from capture.base import CaptureSource`                  |
| Constructeur                   | `__init__(self, target_fps=120)` | Ajout paramètre `source: CaptureSource` (injection externe — option a) |
| Bloc init DXCam dans `start()` | Lignes 16-19                     | Remplacé par `self._source.start(self._target_fps)`                    |
| Bloc stop DXCam dans `stop()`  | Ligne 33                         | Remplacé par `self._source.stop()`                                     |
| Lecture frame dans `_worker()` | Ligne 49 (`get_latest_frame()`)  | Remplacé par `self._source.grab()`                                     |
| Log diagnostic FPS réel        | Ligne 26 (`"FPS réel DXCam"`)    | Remplacé par `f"FPS réel {self._source.name}"`                         |

**Invariants préservés** :

- API publique `start()`, `stop()`, `get_frame()`, `get_frame_id()` — signatures et sémantique inchangées.
- Attributs `_latest_frame`, `_latest_ts`, `_frame_id` — inchangés (sous réserve grep préalable).
- Stop signal `self._running` (flag bool) — conservé tel quel.

**Côté `main.py`** : `SourceSelector.resolve(config)` appelé **avant** `CaptureThread(source=..., target_fps=...)`. `CaptureSourceNotFound` catché en amont (Phase 6).

---

## Phase 6 — Gestion d'erreur et messages utilisateur

**Préconditions** : Phase 5 livrée.

**Exit immédiat (V1)** :

- `CaptureSourceNotFound` levée par `SourceSelector.resolve()`, catchée dans `main.py` **avant** instanciation `CaptureThread`.
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
- Détection de perte de flux runtime (FSE drop post-démarrage) — non couverte en V1, relance script requise.

---

## Phase 7 — Config `config.yaml`

**Préconditions** : Phase 4 livrée.

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
- `obs_virtual_camera_name` non vide (string)

---

## Phase 8 — Ajout MSSSource (additive, post-Phase 7)

**Préconditions** : : Phases 1-5 livrées.

**Objectif** : ajouter une 3e source de capture (MSS) .

**Décisions figées** :

- Validation résolution stricte (Option A — raise si mismatch)
- Sélection moniteur via config.capture.mss_monitor_index (défaut 1)
- target_fps stocké mais inutilisé (capture synchrone)
- Position dans source_priority : à définir en config selon priorité user

## Phase 9 — Documentation

**Préconditions** : Phase 5 livrée.

- `README.md` : mise à jour section **Dépendances** (aucun ajout, `comtypes` déjà présent)
- `README.md` : mise à jour section **Lancement** — prérequis source (Borderless ou OBS Virtual Camera)
- `README.md` : mention du contrat `CaptureSource` A1 strict pour faciliter l'ajout de futures sources.
- `config.yaml` : commentaires inline sur les nouvelles clés
- Note explicite : **bascule DXCam → FSE post-démarrage non gérée en V1**, relance script requise

---

## Séquence d'exécution

| Phase                         | Dépendance     | Durée estimée | Statut    |
| ----------------------------- | -------------- | ------------- | --------- |
| 0 — Audit                     | Aucune         | 30 min        | ✅ Livrée |
| 1 — Abstraction base (A1)     | Phase 0        | 30 min        | ⏳ Prête  |
| 2 — DXCamSource               | Phase 1        | 30 min        | ⏸ Pause   |
| 3 — Cv2Source                 | Phase 1        | 1 h           | ⏸ Pause   |
| 4 — SourceSelector (B1 + β)   | Phases 2 + 3   | 1 h           | ⏸ Pause   |
| 5 — Intégration CaptureThread | Phase 4 + grep | 30 min        | ⏸ Pause   |
| 6 — Gestion erreur            | Phase 5        | 30 min        | ⏸ Pause   |
| 7 — Config                    | Phase 4        | 15 min        | ⏸ Pause   |
| 8 — Doc                       | Phase 5        | 30 min        | ⏸ Pause   |

**Total estimé restant :** 4 h 30.

**Règle de séquentialité** : une phase ne peut démarrer que si toutes ses dépendances sont livrées et validées. Aucun parallélisme implicite.

---

### Ce que ce plan ne change pas

- API publique `CaptureThread` : inchangée
- Tous les threads aval (`DetectThread`, `FastTrackThread`, `SendThread`) : inchangés
- `main.py` : un seul ajout — `SourceSelector.resolve()` + catch `CaptureSourceNotFound` → exit propre
- `config.yaml` : section additive, aucune clé existante modifiée
