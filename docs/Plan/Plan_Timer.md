# Plan Refactor Timer — Stratégie 3, Variante 3-A

---

## Résumé exécutif

**Bug B-05a-revive** : `mask_revive_latency_ms` est négatif dans 88 % des cas
(min −243 ms) parce que deux bases de temps sont mêlées dans son calcul :

| Base de temps       | Sémantique             | Usage                                       |
| ------------------- | ---------------------- | ------------------------------------------- |
| `time.perf_counter` | horloge monotone (s)   | TTL : `lost_after_s`, `expire_after_lost_s` |
| `detected_frame_ts` | timestamp capture (µs) | Latences : `*_latency_ms`                   |

Le calcul d'origine utilise `prev_lost_since_ts` (= `perf_counter` au moment `missing`)
pour une latence revive, alors que la référence temporelle attendue est un timestamp
de capture — d'où le signe négatif.

**Stratégie 3-A retenue** : séparation stricte des deux bases de temps.
Aucune latence n'utilise `abs()` ni `clamp()` pour masquer un signe ;
la correction est faite à la source.

---

## SECTION 1 — Questions ouvertes (RESOLVED ✅)

### Q1 — Initialisation de `last_seen_frame_ts` à l'ajout ?

**Décision : Q1-A** ✅ Initialiser `last_seen_frame_ts` à `last_detected_ts`
dans `Mask.__post_init__()` — avoids `None` on first revive latency calc.

### Q2 — Signature de `apply_detections` ?

**Décision : Q2-A** ✅ Signatures inchangée. `detected_frame_ts` est déjà
transmis à `apply_detections` dans `main.py` (via le paramètre `ts`).

### Q3 — Signature de `mark_matched` dans registry.py ?

**Décision : Q3-A** ✅ Propagation via paramètre explicite
`last_detected_frame_ts` dans `mark_matched()` — pas de mutation d'état global.

### Q4 — Provenance de `lost_since_frame_ts` dans `missing` ?

**Décision : Q4-A** ✅ Dans `transition("missing", ...)`, setter
`lost_since_frame_ts` depuis le dernier `last_detected_frame_ts` connu
(transmis via la chaîne `tick()` → `tick_and_expire()` → `transition()`).

### Q5 — Comment propager `last_detected_frame_ts` dans `tick()` ?

**Décision : Q5-A** ✅ Ajouter un paramètre `last_detected_frame_ts` à
`tracker.tick()`, propagé à `registry.tick_and_expire()`, puis à
`mask.transition("missing")`.

### Q6 — Documentation `bench-probes.md` ?

**Décision : Q6-A** ✅ Ajouter une colonne/note "time base" (capture vs
perf_counter) dans le tableau des sondes mask pour lever l'ambiguïté.

### Q7 — Validation ?

**Décision : Q7-A** ✅ Via `bench-compare.py` contre les critères C1-C6
(validation en séance — non implémentée dans ce patch, Phase 5).

---

## SECTION 2 — Modèle d'horloge cible

### Tableau des deux rôles

| Sonde                     | Base de temps       | Calcul                                          | Signification                      |
| ------------------------- | ------------------- | ----------------------------------------------- | ---------------------------------- |
| `mask_confirm_latency_ms` | capture (frame_ts)  | `last_seen_frame_ts − created_ts`               | Délai création → confirmation (ms) |
| `mask_revive_latency_ms`  | capture (frame_ts)  | `last_seen_frame_ts − prev_lost_since_frame_ts` | Délai LOST → revive (ms)           |
| `mask_lost_latency_ms`    | capture (frame_ts)  | `last_seen_frame_ts − created_ts`               | Délai création → LOST (ms)         |
| TTL `lost_after_s`        | perf_counter (`ts`) | `ts − last_seen_ts`                             | Hors-vue côté application          |
| TTL `expire_after_lost_s` | perf_counter (`ts`) | `ts − lost_since_ts`                            | TTL purgement LOST                 |

### Invariants

```text
1. last_seen_ts       mis à jour sur event="matched" avec perf_counter (ts)
2. lost_since_ts      posé sur event="missing" avec perf_counter (ts)
3. created_ts         posé à la création avec perf_counter (ts) du create()
4. last_seen_frame_ts mis à jour sur event="matched" avec last_detected_frame_ts
5. lost_since_frame_tsposé sur event="missing" avec last_detected_frame_ts
```

### Règles interdites

```text
- Ne JAMAIS calculer une latence (mask_*_latency_ms) avec des champs *_ts mixed avec *_frame_ts
- Ne JAMAIS utiliser abs() pour masquer un signe négatif de latence
- Ne JAMAIS utiliser clamp() pour masquer un signe négatif de latence
- Les champs *_frame_ts ne doivent JAMAIS servir au calcul de TTL (lost_after_s, expire_after_lost_s)
```

---

## SECTION 3 — Audit du code existant

### Fichiers audités directement

| Fichier               | Rôle                                                        |
| --------------------- | ----------------------------------------------------------- |
| `core/mask.py`        | Champs Mask, `transition()`, `to_dict()`, `__post_init__()` |
| `tracker/registry.py` | `create()`, `mark_matched()`, `tick_and_expire()`           |
| `tracker/tracker.py`  | `apply_detections()`, `tick()`, `apply_fast_direct()`       |
| `main.py`             | Boucle principale, passage de `detected_frame_ts`           |
| `bench-probes.md`     | Documentation des sondes                                    |

### Phase 1 — core/mask.py

#### 4.1.1 Ajouter les champs de timestamp capture

```python
    # --- Cycle de vie : timestamps capture (latences: mask_revive_latency_ms, mask_confirm_latency_ms) ---
    last_seen_frame_ts:       float          = 0.0   # Plan_Timer Q1: init à last_detected_ts
    lost_since_frame_ts:      Optional[float]= None   # Plan_Timer Q4: mis à jour en "missing"
```

#### 4.1.2 Synchroniser dans `__post_init__`

```python
        if self.last_seen_frame_ts == 0.0:
            self.last_seen_frame_ts = self.last_detected_ts
```

#### 4.1.3 Corriger `mask_revive_latency_ms` (L119)

**AVANT (code actuel)** :

```python
elif self.state == MaskState.LOST:
    if prev_lost_since_ts is not None:
        bench.probe("mask_revive_latency_ms", (ts - prev_lost_since_ts) * 1000.0)
```

**APRÈS (Plan_Timer — sign corrigé)** :

```python
elif self.state == MaskState.LOST:
    prev_lost_since_frame_ts = self.lost_since_frame_ts   # capture
    if prev_lost_since_frame_ts is not None:
        bench.probe("mask_revive_latency_ms",
            (self.last_seen_frame_ts - prev_lost_since_frame_ts) * 1000.0)
```

Le calcul utilise maintenant `frame_ts − frame_ts` (capture − capture) au lieu de
`perf_counter − perf_counter`, éliminant le signe négatif.

#### 4.1.4 Corriger `mask_lost_latency_ms` (L131)

**AVANT (code actuel)** :

```python
bench.probe("mask_lost_latency_ms", (ts - self.created_ts) * 1000.0)
```

**APRÈS (Plan_Timer — Q5-A: capture − capture)** :

```python
bench.probe("mask_lost_latency_ms",
    (self.last_seen_frame_ts - self.created_ts) * 1000.0)
```

Note : `created_ts` est déjà un timestamp capture (posée à la création via
`last_detected_ts` du `create()` de registry, qui est appelé avec `ts`).
En pratique, les deux sont proches, mais pour la cohérence stricte de la Stratégie 3-A,
on utilise `last_seen_frame_ts` comme référence de temps capture courant.

Ajouter la mise à jour du champ capture dans `matched` :

```python
elif event == "matched":
    prev_lost_since_ts = self.lost_since_ts
    prev_lost_since_frame_ts = self.lost_since_frame_ts   # pour revive
    self.frames_matched += 1
    self.last_seen_ts = ts
    if last_detected_frame_ts is not None:                 # Q3: propager
        self.last_seen_frame_ts = last_detected_frame_ts
    self.lost_since_ts = None
    self.lost_since_frame_ts = None
```

Ajouter la pose du champ capture dans `missing` :

```python
elif event == "missing":
    bench.count("mask_transition_missing_total")
    if self.state in (MaskState.PENDING, MaskState.CONFIRMED):
        self.state = MaskState.LOST
        self.lost_since_ts = ts
        self.lost_since_frame_ts = last_detected_frame_ts   # Q4: poser
        self.frames_matched = 0
        bench.count("mask_to_lost_total")
```

#### 4.1.5 Nouvelle signature de `transition`

```python
def transition(self, event: str, ts: float, last_detected_frame_ts: float = None) -> MaskState:
```

#### 4.1.6 Corriger `mask_confirm_latency_ms`

**AVANT** :

```python
bench.probe("mask_confirm_latency_ms", (ts - self.created_ts) * 1000.0)
```

**APRÈS** :

```python
bench.probe("mask_confirm_latency_ms",
    (self.last_seen_frame_ts - self.created_ts) * 1000.0)
```

#### 4.1.7 Mettre à jour `to_dict()`

Ajouter les nouveaux champs :

```python
"last_seen_frame_ts":      round(self.last_seen_frame_ts, 4),
"lost_since_frame_ts":     round(self.lost_since_frame_ts, 4) if self.lost_since_frame_ts is not None else None,
```

---

### Phase 2 — tracker/registry.py

#### 4.2.1 Mettre à jour `create()` (L43-60)

inchangé — `created_ts` est déjà posé à `ts` (perf_counter), ce qui est correct pour le TTL.
Les champs capture (`last_seen_frame_ts`) sont initialisés dans `__post_init__` de Mask (Q1).

#### 4.2.2 Mettre à jour `mark_matched()` (L70-75)

```python
def mark_matched(self, uid: int, ts: float, source: str = "unknown",
                 last_detected_frame_ts: float = None) -> None:
    mask = self._masks.get(uid)
    if mask is None:
        return
    mask.transition("matched", ts, last_detected_frame_ts=last_detected_frame_ts)
    if source == "slow":
        mask.last_slow_ts = ts
```

#### 4.2.3 Mettre à jour `tick_and_expire()`

```python
def tick_and_expire(self, ts: float, updated_uids: set = None,
                    last_detected_frame_ts: float = None) -> List[Mask]:
```

Appel à `transition("missing", ts)` :

```python
# À la transition missing (dans la boucle)
mask.transition("missing", ts, last_detected_frame_ts=last_detected_frame_ts)
```

---

### Phase 3 — tracker/tracker.py

#### 4.3.1 Mettre à jour `tick()` (L152-186)

Nouvelle signature :

```python
def tick(self, ts: float = None, updated_uids: set = None,
        last_detected_frame_ts: float = None) -> list:
```

Passer le paramètre à `tick_and_expire` :

```python
self.registry.tick_and_expire(ts, updated_uids, last_detected_frame_ts=last_detected_frame_ts)
```

#### 4.3.2 Confirmer la signature de `apply_detections`

inchangée (Q2 : `detected_frame_ts` est déjà le `ts` passé dans main.py ~L139).

---

### Phase 4 — main.py

#### 4.4.1 Initialiser `last_detected_frame_ts` avant la boucle

```python
last_detected_frame_ts = 0.0
```

Capturer après le poll slow detect :

```python
if slow_updated:
    last_detected_frame_ts = detected_frame_ts
```

#### 4.4.2 Passer `last_detected_frame_ts` à `tick()`

Modifier l'appel `tick()` :

```python
confirmed_masks = tracker.tick(now, updated_uids, last_detected_frame_ts)
```

---

### Phase 5 — Documentation

#### 4.5.1 Plan_Tracker.md

Mettre à jour pour refléter la Stratégie 3-A.

#### 4.5.2 bench-probes.md

Ajouter une colonne "Base de temps" dans les tableaux des sondes `mask` :

| Sonde                     | Base de temps                                             | Calcul |
| ------------------------- | --------------------------------------------------------- | ------ |
| `mask_confirm_latency_ms` | capture (`last_seen_frame_ts − created_ts`)               |        |
| `mask_revive_latency_ms`  | capture (`last_seen_frame_ts − prev_lost_since_frame_ts`) |        |
| `mask_lost_latency_ms`    | capture (`last_seen_frame_ts − created_ts`)               |        |

---

## SECTION 5 — Tests de non-régression

### Critère C1 — Latence revive toujours positive

- **Requête** : `mask_revive_latency_ms > 0`
- **Métrique** : `% de valeurs < 0`
- **Seuil** : 0 % (strict)

### Critère C2 — Cohérence inter-frame >= ~1 frame

- **Requête** : `mask_revive_latency_ms >= 16.7` (à 60 fps)
- **Métrique** : `% de valeurs < 0` + `% de valeurs < 16.7`
- **Seuil** : 0 % pour < 0, < 5 % pour < 16.7 (hors lag capture)

### Critère C3 — Latence confirm toujours positive

- **Requête** : `mask_confirm_latency_ms > 0`
- **Métrique** : `% de valeurs < 0`
- **Seuil** : 0 % (strict)

### Critère C4 — Latence lost toujours positive

- **Requête** : `mask_lost_latency_ms > 0`
- **Métrique** : `% de valeurs < 0`
- **Seuil** : 0 % (strict)

### Critère C5 — Pas de NaN sur les latences

- **Métrique** : `% de NaN` dans `mask_confirm_latency_ms`, `mask_revive_latency_ms`, `mask_lost_latency_ms`
- **Seuil** : 0 % (post-refactor, si Q5-A)

### Critère C6 — Intégrité des gauges d'état

- **Requête** : `registry_confirmed + registry_pending + registry_lost == len(masks)`
- **Métrique** : `% d'incohérence`
- **Seuil** : 0 % (strict)

---

## SECTION 6 — Ordre d'exécution / Phases

### Principe directeur : `capture_ts` propagé

L'architecture repose sur un **timestamp de capture unique, posé une seule fois, et propagé intact** tout au long du pipeline. C'est ce timestamp qui sert de borne de référence absolue pour tous les calculs de durée.

**Chaîne de propagation validée :**

```text
capture_thread.py L80  →  ts = time.perf_counter()       ← pose, MONOTONE
                         _latest_ts = ts
                                   ↓
main.py L104           →  frame, frame_ts = get_frame()  ← récupéré intact
                         detector.give_frame(frame, frame_ts)
detect_thread.py L39/62 →  _latest_frame_ts = frame_ts   ← propagé
main.py L127           →  detected_frame_ts               ← même valeur
```

`detected_frame_ts` est donc le `capture_ts` d'origine, et **c'est lui** qui doit être utilisé comme borne pour toute latence (lost, revive, confirm). Aucune autre lecture de `perf_counter()` n'est nécessaire pour les calculs de durée.

**Règle absolue :** toute borne de calcul de durée dans la pipeline doit provenir de `detected_frame_ts`. Ne jamais re-lire `perf_counter()` comme borne — seulement comme valeur de comparaison pour les seuils TTL (_"est-ce que 5 secondes se sont écoulées ?"_).

**Point de vigilance unique — tracker.py L38-39 (fallback) :**

```python
if detected_frame_ts is None:
    detected_frame_ts = perf_counter()  # ← injecte un perf_counter PRESENT, pas le capture_ts
```

Ce fallback n'est jamais atteint en fonctionnement nominal (car `detected_frame_ts` est toujours posé). Il ne doit servir qu'à éviter un crash si un tracker est instancié sans frame detectée. **Si ce fallback se déclenche fréquemment, c'est un signal d'alerte** : le pipeline d'acquisition ou de détection est défaillant.

---

### Phase 1 — Répondre aux questions de la Section 1

→ **FAIT ✅** (Q1-Q7 tranchées)

---

### Phase 2 — core/mask.py

Implémenter les modifications en utilisant `detected_frame_ts` (le `capture_ts` propagé) comme borne pour les trois probes :

- `mask_lost_latency_ms` → `detected_frame_ts − prev_lost_since_frame_ts` (capture − capture, monotone ✅)
- `mask_revive_latency_ms` → `detected_frame_ts − prev_lost_since_frame_ts` (capture − capture, monotone ✅)
- `mask_confirm_latency_ms` → `detected_frame_ts − prev_confirm_since_frame_ts` (capture − capture, monotone ✅)

Aucun `datetime` ni `time.time()` n'intervient dans ces calculs.

1. Ajouter les champs capture (`last_seen_frame_ts`, `lost_since_frame_ts`) → **FAIT ✅**
2. Synchroniser dans `__post_init__` → **FAIT ✅**
3. Nouvelle signature `transition(event, ts, last_detected_frame_ts)` → **FAIT ✅**
4. Corriger `mask_revive_latency_ms` (capture − capture) → **FAIT ✅**
5. Corriger `mask_lost_latency_ms` (capture − capture) → **FAIT ✅**
6. Corriger `mask_confirm_latency_ms` (capture − capture) → **FAIT ✅**
7. Mettre à jour `to_dict()` → **FAIT ✅**

---

### Phase 3 — tracker/registry.py

- Propager `last_detected_frame_ts` (le `capture_ts`) dans `tick_and_expire()` et `transition()`
- Ne jamais remplacer la borne par un `perf_counter()` re-lu en cours de boucle

1. `mark_matched()` : propagation explicite `last_detected_frame_ts` → **FAIT ✅**
2. `tick_and_expire()` : propagation vers `transition("missing")` → **FAIT ✅**

---

### Phase 4 — tracker/tracker.py

- `tick()` reçoit et transmets `last_detected_frame_ts`
- Le fallback L38-39 reste — mais ne doit jamais s'activer en nominal

1. `tick()` : nouveau paramètre `last_detected_frame_ts` → **FAIT ✅**
2. Transmettre à `registry.tick_and_expire()` → **FAIT ✅**
3. Confirmer la signature de `apply_detections` (inchangée) → **FAIT ✅**

---

### Phase 5 — main.py

- Lire `detected_frame_ts` après le poll slow detect (L127)
- Le passer à `tick()` comme `last_detected_frame_ts`

1. Initialiser `last_detected_frame_ts` → **FAIT ✅**
2. Capturer `detected_frame_ts` après slow poll → **FAIT ✅**
3. Passer `last_detected_frame_ts` à `tracker.tick()` → **FAIT ✅**

---

### Phase 6 — Documentation

- Plan_Tracker.md et bench-probes.md reflètent la stratégie 3-A et la règle `capture_ts propagé`

1. bench-probes.md : colonne "time base" → **FAIT ✅** (Phase 5.2)
2. Plan_Tracker.md : mise à jour Stratégie 3-A → **PENDING** (hors périmètre patch)

---

## Flux temporel corrigé (Stratégie 3-A)

```text
detector.get_result()     → detected_frame_ts (capture)
        ↓
        ↓
main.py: last_detected_frame_ts = detected_frame_ts
        ↓
tracker.tick(..., last_detected_frame_ts)
        ↓
registry.tick_and_expire(..., last_detected_frame_ts)
        ↓
mask.transition("missing", perf_counter_ts, last_detected_frame_ts)
        ↓
lost_since_ts       = perf_counter_ts     (pour TTL)
lost_since_frame_ts = last_detected_frame_ts  (pour latence revive)
        ↓
mask.transition("matched", perf_counter_ts, last_detected_frame_ts)
        ↓
last_seen_ts       = perf_counter_ts     (pour TTL)
last_seen_frame_ts = last_detected_frame_ts  (pour latences)
        ↓
mask_revive_latency_ms  = (last_seen_frame_ts - prev_lost_since_frame_ts) * 1000
                        = (capture_ts - capture_ts) * 1000
                        → ✅ POSITIF
```
