# Plan Refactor Timer — Stratégie 3, Variante 3-A

> **Statut** : Plan d'implémentation — en attente de résolution des questions Section 1
> **Projet** : Rocket League nameblur — pipeline de tracking avec floutage de plaques
> **Variante retenue** : Stratégie 3-A — TTL en `perf_counter`, latences en capture-capture
> **Bug d'origine** : B-05a-revive (`mask_revive_latency_ms` négatif, min −243 ms, 88 % des revives)

---

## Résumé exécutif

Le refactor sépare rigoureusement les deux rôles d'horloge dans le pipeline de tracking :

- **Rôle "Durée écoulée"** → `time.perf_counter()` (monotone, avance à chaque frame) : seuil de TTL `lost_after_s` et `expire_after_lost_s` uniquement.
- **Rôle "Instant métier"** → `detected_frame_ts` (horodatage capture caméra, propagé de main.py) : latences métier `mask_confirm_latency_ms`, `mask_revive_latency_ms`, `mask_lost_latency_ms` uniquement.

Variante 3-A : **aucune modification de la sémantique des TTL**. `perf_counter` reste la seule base pour les comparaisons de seuils. Les latences passent en capture-capture en ajoutant deux champs de timestamp sur le `Mask` (`last_seen_frame_ts`, `lost_since_frame_ts`), propagés depuis `detected_frame_ts`. Le `ts` du "missing" (dû à l'absence de détection cette frame) utilise le **dernier `detected_frame_ts` connu** (sémantique (i)), persisté dans `main.py` et descendu dans `tick` → `tick_and_expire` → `transition`.

Le clamp ou `abs()` sur `mask_revive_latency_ms` est explicitement interdit : il masque le bug au lieu de le corriger.

---

## SECTION 1 — Questions à préciser AVANT réécriture

> Chaque question bloque une partie du plan. Répondre à toutes avant de toucher au code.

---

### Q1 — `detected_frame_ts` disponible dans `tick_and_expire` via quel chemin ?

**Énoncé** : Dans main.py (L141), `tracker.apply_detections(..., detected_frame_ts, "slow")` reçoit `detected_frame_ts` retourné par `detector.get_result()`. Cependant, `tick(now, updated_uids)` (L161) reçoit `now = time.perf_counter()`. `tick_and_expire()` est appelé depuis `tick()` et ne reçoit donc aucun `detected_frame_ts`.

**Pourquoi ça bloque** : Pour la sémantique (i) ("missing" à la frame N → dater du dernier `detected_frame_ts` connu), `tick_and_expire` doit connaître ce dernier timestamp. Soit on passe un paramètre additionnel, soit on expose un accesseur sur le Tracker (ou le DetectThread).

**Options pressenties** :

- **Q1-A** : Passer `detected_frame_ts` comme 3ᵉ paramètre à `tick(ts, updated_uids, detected_frame_ts=None)` et le redispatcher à `tick_and_expire`.
- **Q1-B** : Stocker le dernier `detected_frame_ts` dans `Tracker` (via `apply_detections` qui le reçoit) et lire `Tracker._last_detected_frame_ts` depuis `tick_and_expire` — plus simple pour la propagation mais introduit un état mutable sur le Tracker.
- **Q1-C** : Maintenir une variable locale dans `main.py` (avant la boucle `while`) et la passer directement à `tick(ts, updated_uids, last_detected_frame_ts)` à chaque itération.

**Decision requise** : Choisir Q1-A, Q1-B ou Q1-C.

---

### Q2 — Signature de `apply_detections` dans tracker.py

**Énoncé** : D'après l'audit, `apply_detections` dans tracker.py est invoqué en L141 de main.py avec `(frame, dets, detected_frame_ts, "slow")`. Cependant, `apply_detections` ne figure pas dans le code visible de tracker.py (227 lignes) — son implémentation est peut-être dans un module non listé dans les uploads, ou les lignes pertinentes sont tronquées.

**Pourquoi ça bloque** : Si on modifie les signatures dans `apply_detections` ou `mark_matched`, il faut connaître la signature actuelle exacte (n° de paramètres, noms des paramètres, type de `source`).

**Options pressenties** :

- **Q2-A** : Considérer que `apply_detections` ne prend que `(frame, detections, ts, source)` et que `ts` est déjà utilisé comme `detected_frame_ts`.
- **Q2-B** : Ouvrir tracker.py au complet pour obtenir la signature exacte avant d'écrire le plan de modification.

**Decision requise** : Confirmer Q2-A ou ouvrir le fichier.

---

### Q3 — Signature de `mark_matched` dans registry.py

**Énoncé** : `mark_matched()` existe en registry.py (L70-75) et prend `(uid, source, ts)`. La question est si son paramètre `ts` est effectivement utilisé pour le `transition("matched", ts)` et pour `mask.last_slow_ts = ts` (ligne 75).

**Pourquoi ça bloque** : Si `mark_matched` est appelé depuis le fast tracker avec `perf_counter` (comme `mark_matched(uid, "fast", now)`), alors les champs `_frame_ts` requis par la Stratégie 3-A doivent aussi être posés dans `mark_matched`.

**Options pressenties** :

- **Q3-A** : `mark_matched` reçoit déjà `ts` qui est du `detected_frame_ts` pour le fast (fourni par le thread fast tracker). → Ajouter `last_seen_frame_ts` dans le même appel.
- **Q3-B** : `mark_matched` reçoit `now` (perf_counter) et doit être modifié pour recevoir aussi `frame_ts`. → Modifier la signature de `mark_matched` et de tous ses appelants.

**Decision requise** : Q3-A ou Q3-B.

---

### Q4 — Champs manquants sur `Mask` : `confirm_after`, `lost_after_s` viennent-ils de la config ?

**Énoncé** : Le `Mask` dataclass (mask.py L80-82) porte des champs `confirm_after`, `lost_after_s` avec des valeurs par défaut (2 et 1.0). Le `registry.add()` (registry.py L52-53) passe aussi `confirm_after=self.cfg.confirm_after`. Le TTL utilise `mask.lost_after_s` (registry.py L96).

**Pourquoi ça bloque** : Les TTL themselves restent en `perf_counter` dans la Stratégie 3-A — aucun changement sur ce mécanisme. Cependant, il faut confirmer que le champ `lost_after_s` sur le `Mask` est bien initialisé depuis la config pour que le refactor n'affecte pas les comportements de TTL.

**Options pressenties** :

- **Q4-A** : Confirmer que `registry.add()` initialise bien `lost_after_s` depuis la config (comme il le fait pour `confirm_after`). Si non, l'ajouter.
- **Q4-B** : Vérifier si `lost_after_s` est copié depuis la config à la création du Mask.

**Decision requise** : Confirmer Q4-A ou corriger.

---

### Q5 — `mask_lost_latency_ms` : mixer ou non après le refactor ?

**Énoncé** : Dans la situation actuelle, `mask_lost_latency_ms = (ts − created_ts) * 1000.0` est **homogène** (mask.py L131) car les deux opérandes sont en `detected_frame_ts` (création = `add()` en capture, transition LOST = `transition("missing", now)` en perf_counter... sauf que `transition` est appelé depuis `tick_and_expire` avec `ts = now`). **Tous les paramètres de `transition("missing", ts)` viennent de `perf_counter`**. Donc `mask_lost_latency_ms` **n'est PAS homogène aujourd'hui** : `created_ts` (capture) − `ts` du "missing" (perf_counter).

`detected_frame_ts` (capture, en retard de ~240 ms). Ce qui rendrait le résultat systématiquement négatif ou falsifié, sauf si les détections sont assez rapprochées pour que le drift reste faible.

Avec la Stratégie 3-A, si `transition("missing", last_detected_frame_ts)` est appelé (sémantique (i)), alors `mask_lost_latency_ms = last_detected_frame_ts − created_ts` : **homogène capture-capture**. Si on utilise la sémantique (ii) (nan/none), la sonde disparaît du flux sur les "missing".

**Pourquoi ça bloque** : Le comportement attendu de `mask_lost_latency_ms` après refactor n'est pas encore formellement décidé.

**Options pressenties** :

- **Q5-A** : `mask_lost_latency_ms` devient capture-capture (`last_detected_frame_ts − created_ts`). La sonde est émise sur chaque transition "missing". Valeur typique ≈ temps entre dernière détection et perte de vue.
- **Q5-B** : `mask_lost_latency_ms` est émise seulement si un `detected_frame_ts` est disponible (sémantique (i)-b). Sinon absente du flux cette frame.

**Decision requise** : Q5-A ou Q5-B.

---

### Q6 — Mise à jour de `bench-probes.md` ?

**Énoncé** : `bench-probes.md` (L195-200) documente les 3 sondes latences avec des descriptions exactes mais **sans préciser la base de temps** des timestamps utilisés. Exemple : `mask_revive_latency_ms` → "Délai entre entrée LOST et revive (ms)". Le choix de base capture-capture après le refactor doit être explicité pour documenter le contrat.

**Pourquoi ça bloque** : Sans mise à jour de `bench-probes.md`, le contrat temporel des sondes reste ambigu et un futur développeur pourrait reproduire le bug.

**Options pressenties** :

- **Q6-A** : Mettre à jour `bench-probes.md` dans la Phase 3 (Plan_Tracker.md) avec la colonne "Base de temps" explicitant capture/capture pour les 3 sondes latences.
- **Q6-B** : Laisser `bench-probes.md` inchangé (les descriptions suffisent) — la documentation de base de temps appartient au Plan_Timer et Plan_Tracker.

**Decision requise** : Q6-A ou Q6-B.

---

### Q7 — Champs `last_seen_frame_ts` / `lost_since_frame_ts` : initialisation par défaut

**Énoncé** : Les deux nouveaux champs à ajouter au `Mask` sont des `float` avec une sémantique "capture". Valeur par défaut ?

**Pourquoi ça bloque** : `last_seen_frame_ts` doit être initialisé à la création du `Mask` (dans `registry.add()`). `lost_since_frame_ts` est `Optional[float]` (comme `lost_since_ts`) initialisé à `None`.

**Options pressenties** :

- **Q7-A** : `last_seen_frame_ts: float = 0.0` dans le dataclass, avec `__post_init__` qui le pose à `last_detected_ts` si non fourni.
- **Q7-B** : `last_seen_frame_ts: float` sans défaut, rendu obligatoire à la création (forcé par `registry.add()`).
- **Q7-C** : `last_seen_frame_ts: float = 0.0`, `__post_init__` qui synchonise avec `last_seen_ts` en attendant le premier match.

**Decision requise** : Q7-A, Q7-B ou Q7-C.

---

## SECTION 2 — Modèle d'horloge cible

### Tableau des deux rôles

| Rôle                          | Horloge               | Source                                                         | Usage                                                                                      | Maintien                                                                |
| ----------------------------- | --------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **Durée écoulée** (TTL)       | `time.perf_counter()` | `now` posé en début de frame (main.py L100)                    | Comparaison `ts − last_seen_ts ≥ lost_after_s`, `ts − lost_since_ts ≥ expire_after_lost_s` | **Invariable** — `perf_counter` avance chaque frame même sans détection |
| **Instant métier** (Latences) | `detected_frame_ts`   | `frame_ts` retourné par `detector.get_result()` (main.py L127) | Tous les `bench.probe` latences, tous les champs `_frame_ts` sur le `Mask`                 | Induit un lag visible : `capture = now − ~pipeline_latency`             |

### Invariants

1. `perf_counter` ne doit **jamais** être soustrait à un `detected_frame_ts`, ni l'inverse.
2. Les champs `_ts` (suffixe sans `_frame`) restent en `perf_counter` pour les TTL uniquement.
3. Les champs `_frame_ts` sont en `detected_frame_ts` uniquement.
4. `transition("matched", ts)` : le `ts` de la sonde est en **capture** ; les TTL comparison utilisent `last_seen_ts` en **perf_counter**.
5. `transition("missing", ts)` : le `ts` de la sonde est en **capture** (sémantique (i)) ou absent (sémantique (ii)).

### Règles interdites

- `abs()` ou `clamp()` sur une valeur de latence pour la rendre positive — c'est un anti-pattern qui masque le bug.
- Mélange de `perf_counter` et `detected_frame_ts` dans une même soustraction.
- `perf_counter` comme valeur de `lost_since_frame_ts` ou `last_seen_frame_ts`.

---

## SECTION 3 — Audit du code existant

### Fichiers audités directement

| Fichier                      | Éléments timestamp trouvés                                                                                                             | Base actuelle                 | Action requise                                                   | Statut                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------- | ------------------------- |
| **main.py**                  | `now = time.perf_counter()` L100 (top de frame) · `detected_frame_ts` obtenu L127 via `detector.get_result()`                          | `now` = perf_counter          | Passer `detected_frame_ts` à `tick()`                            | ✅ Audité                 |
|                              | `tracker.apply_detections(frame, dets, detected_frame_ts, "slow")` L141 · `tracker.tick(now, updated_uids)` L161                       | `detected_frame_ts` = capture | persister le dernier `detected_frame_ts` entre frames            | ✅ Audité                 |
| **core/mask.py**             | `last_seen_ts` L76 (perf_counter) · `lost_since_ts` L77 (perf_counter) · `created_ts` L78 (capture — posé par `add()`)                 | Mixte                         | Ajouter `last_seen_frame_ts` et `lost_since_frame_ts`            | ✅ Audité                 |
|                              | `mask_confirm_latency_ms` L116 : `(ts − created_ts)` → capture−capture ✅ ·                                                            |                               | corriger les 2 sondes défectueuses (L119, L131)                  | ✅ Audité                 |
|                              | `mask_revive_latency_ms` L119 : `(ts − prev_lost_since_ts)` → capture − perf_counter ❌ ·                                              |                               |                                                                  | ✅ Audité                 |
|                              | `mask_lost_latency_ms` L131 : `(ts − created_ts)` → perf_counter − capture ❌                                                          |                               |                                                                  | ✅ Audité                 |
| **tracker/registry.py**      | `registry.add()` L48 : `last_detected_ts=ts, last_seen_ts=ts, created_ts=ts` (ts=capture) ✅                                           | Mixte                         | Ajouter propagation de `last_seen_frame_ts` dans `add()`         | ✅ Audité                 |
|                              | `mark_matched()` L70-75 : `transition("matched", ts)` · `tick_and_expire()` L84 : `ts = perf_counter` (reçu de `tick()`) ·             |                               | et `mark_matched()` `tick_and_expire` doit                       | ✅ Audité                 |
|                              | TTL comparison L96-97 : `ts − last_seen_ts` en perf_counter ✅ · L102 : `ts − lost_since_ts` en perf_counter ✅                        |                               | recevoir `last_detected_frame_ts` et poser `lost_since_frame_ts` | ✅ Audité                 |
| **tracker/tracker.py**       | `apply_detections()` invoqué L141 de main.py avec `detected_frame_ts`                                                                  | perf_counter et               | Signature `tick(ts, updated_uids, last_detected_frame_ts=None)`  | ⚠️ Parties non visibles   |
|                              | `mark_matched()` invoqué depuis fast tracker (non visible dans le fichier uploadé) · `tick()` L152 :                                   | capture                       | propager à `registry.tick_and_expire()`                          | dans le fichier uploadé   |
|                              | `ts = perf_counter` (défaut si None) `predict_position` appelé pour non-matchés                                                        |                               |                                                                  | à confirmer à l'ouverture |
| **threads/detect_thread.py** | `_latest_frame_ts` L13 (capture) · `_latest_frame_ts_detected` L16 (non utilisé dans le worker visible) ·                              | Capture                       | Confirmer que `_latest_frame_ts_detected`                        | ✅ Audité                 |
|                              | `give_frame(frame, ts)` L35 : stocke `frame_ts` · `get_result()` L41 : retourne les zones ·                                            | Capture                       | (L16) est bien le `detected_frame_ts`                            | ✅ Audité                 |
|                              | `_worker()` L45-63 : compare `last_processed_ts` (avec `frame_ts` capture)                                                             | Capture                       | propagé au tracker, ou s'il existe un autre champ                | ✅ Audité                 |
| **config.yaml**              | `tracker.lifecycle.lost_after_s: 1.0` · `tracker.lifecycle.expire_after_lost_s: 3.0`                                                   | N/A                           | Aucune modification                                              | ✅ Audité                 |
| **Plan_Tracker.md**          | Document de spécification du pipeline                                                                                                  | N/A                           | Mettre à jour avec la Stratégie 3-A et la liste des changements  | ✅ Audité                 |
| **bench-probes.md**          | `mask_confirm_latency_ms` L195 (description) · `mask_revive_latency_ms` L197 (description) · `mask_lost_latency_ms` L200 (description) | N/A                           | Ajouter mention "base de temps capture" dans les descriptions    | ✅ Audité                 |

### Phase 1 — core/mask.py

#### 4.1.1 Ajouter les champs de timestamp capture

Dans le dataclass `Mask`, après la ligne 77 (`lost_since_ts`) :

```python
    # --- Cycle de vie : timestamps (capture) ---
    last_seen_frame_ts:    Optional[float] = None   # capture — pour latences métier
    lost_since_frame_ts:   Optional[float] = None   # capture — pour revive latency
```

#### 4.1.2 Synchroniser dans `__post_init__`

Après la ligne 91 (synchronisation `last_seen_ts / created_ts`) :

```python
        if self.last_seen_frame_ts is None:
            self.last_seen_frame_ts = self.last_detected_ts
```

#### 4.1.3 Corriger `mask_revive_latency_ms` (L119)

**Actuel** :

```python
bench.probe("mask_revive_latency_ms", (ts - prev_lost_since_ts) * 1000.0)
```

**Correction** :

```python
if prev_lost_since_ts is not None:
    bench.probe("mask_revive_latency_ms",
                (ts - self.lost_since_frame_ts) * 1000.0)
```

Note : `lost_since_frame_ts` est posé au moment du "missing" et lu au moment du revive. Toujours en capture. Aucune soustraction cross-base.

#### 4.1.4 Corriger `mask_lost_latency_ms` (L131)

**Actuel** :

```python
bench.probe("mask_lost_latency_ms", (ts - self.created_ts) * 1000.0)
```

**Correction (Q5-A)** :

```python
bench.probe("mask_lost_latency_ms",
            (ts - self.created_ts) * 1000.0)
```

> **Note** : `mask_lost_latency_ms` utilise déjà `created_ts` (capture) et le `ts` du "missing" qui, après refactor, sera le `last_detected_frame_ts`. Donc `created_ts` (capture) − `last_detected_frame_ts` (capture) = homogène capture-capture. Aucune correction de code nécessaire pour L131 si `transition("missing", last_detected_frame_ts)` est respecté.

#### 4.1.5 Mettre à jour `transition("matched")` (L110)

Ajouter la mise à jour du champ capture :

```python
elif event == "matched":
    prev_lost_since_ts = self.lost_since_ts
    self.frames_matched += 1
    self.last_seen_ts = ts                    # perf_counter — TTL
    self.last_seen_frame_ts = ts               # capture — latences
    self.lost_since_ts = None
    self.lost_since_frame_ts = None
```

#### 4.1.6 Mettre à jour `transition("missing")` (L126-128)

Ajouter la pose du champ capture :

```python
elif event == "missing":
    bench.count("mask_transition_missing_total")
    if self.state in (MaskState.PENDING, MaskState.CONFIRMED):
        if (ts - self.last_seen_ts) >= self.lost_after_s:
            self.state = MaskState.LOST
            self.lost_since_ts = ts             # perf_counter — TTL
            self.lost_since_frame_ts = ts        # capture — latences
            self.frames_matched = 0
            bench.count("mask_to_lost_total")
            bench.probe("mask_lost_latency_ms", (ts - self.created_ts) * 1000.0)
```

> Les deux lignes `lost_since_ts` et `lost_since_frame_ts` sont posées **ensemble** avec la même valeur `ts`. C'est acceptable : les deux champs représentent le même instant, converti dans les deux bases. Le TTL utilise `lost_since_ts` (perf_counter), la sonde utilise `lost_since_frame_ts` (capture).

#### 4.1.7 Mettre à jour `to_dict()`

Ajouter les nouveaux champs :

```python
"last_seen_frame_ts": round(self.last_seen_frame_ts, 4) if self.last_seen_frame_ts is not None else None,
"lost_since_frame_ts": round(self.lost_since_frame_ts, 4) if self.lost_since_frame_ts is not None else None,
```

---

### Phase 2 — tracker/registry.py

#### 4.2.1 Mettre à jour `add()` (L43-60)

Ajouter `last_seen_frame_ts=ts` dans l'appel au constructeur Mask :

```python
mask = Mask(
    ...
    last_seen_ts=ts,
    last_seen_frame_ts=ts,      # ← nouveau : posé en capture (identique à last_detected_ts)
    created_ts=ts,
    ...
)
```

#### 4.2.2 Mettre à jour `mark_matched()` (L70-75)

Signature actuelle (?) : `mark_matched(self, uid: int, source: str, ts: float)` — à confirmer (Q3).

**Si Q3-A** (déjà en capture) : ajouter `last_seen_frame_ts` :

```python
mask.last_seen_frame_ts = ts
```

**Si Q3-B** (doit être modifié) : ajouter paramètre `frame_ts` et le propager.

#### 4.2.3 Mettre à jour `tick_and_expire()`

Signature actuelle : `tick_and_expire(self, ts: float, updated_uids: set = None)`
Nouvelle signature : `tick_and_expire(self, ts: float, updated_uids: set = None, last_detected_frame_ts: float = None)`

```python
def tick_and_expire(self, ts: float, updated_uids: set = None,
                   last_detected_frame_ts: float = None) -> List[Mask]:
    """
    last_detected_frame_ts: dernier timestamp capture obtenu par detector.get_result().
                            Utilisé pour dater les transitions "missing" (sémantique (i)).
                            Si None → pas de détection cette frame → utiliser la sémantique (ii).
    """
```

Appel à `transition("missing", ts)` :

```python
# À la ligne ~97 (transition missing)
ts_for_missing = last_detected_frame_ts if last_detected_frame_ts is not None else ts
mask.transition("missing", ts_for_missing)
```

Note : `ts_for_missing` est en **capture** (sémantique (i)) ou en **perf_counter** si aucune détection n'a eu lieu (sémantique (ii)). Cette distinction est la clé du bug B-05a-revive.

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
self.registry.tick_and_expire(ts, updated_uids, last_detected_frame_ts)
```

#### 4.3.2 Confirmer la propagation dans `apply_detections()`

À confirmer (Q2) : si `apply_detections` met déjà à jour un champ `last_detected_frame_ts` sur le Tracker, s'assurer que ce champ n'est pas réécrit par `tick()`.

---

### Phase 4 — main.py

#### 4.4.1 Persister `detected_frame_ts` entre les frames

Déclarer avant la boucle `while` (après L94 `fps_timer`) :

```python
last_detected_frame_ts = 0.0
```

À chaque frame, après `detector.get_result()` (L127), mettre à jour :

```python
last_detected_frame_ts = detected_frame_ts
```

#### 4.4.2 Passer `last_detected_frame_ts` à `tick()`

Modifier l'appel L161 :

```python
confirmed_masks = tracker.tick(now, updated_uids, last_detected_frame_ts)
```

---

### Phase 5 — Documentation

#### 4.5.1 Plan_Tracker.md

Mettre à jour pour refléter la Stratégie 3-A :

- Référencer ce document (`Plan_Timer.md`) comme référence temporelle.
- Lister les fichiers modifiés et les changements de signature.
- Ajouter une section "Modèle d'horloge — Stratégie 3-A" basée sur la Section 2 de ce document.

#### 4.5.2 bench-probes.md

Option Q6-A : ajouter une colonne ou note "Base de temps" dans le tableau du domaine `mask` (L186-200) :

- `mask_confirm_latency_ms` : capture − capture
- `mask_revive_latency_ms` : capture − capture (`lost_since_frame_ts`)
- `mask_lost_latency_ms` : capture − capture (`last_detected_frame_ts − created_ts`)

---

## SECTION 5 — Tests de non-régression

### Critère C1 — `mask_revive_latency_ms` >= 0 à 100 % (B-05a-revive)

- **Métrique** : `% de valeurs < 0` dans `logs/json/bench_agg_*.jsonl` sur une session de référence (avec détections slow cadence 1-2 Hz, revival de masks).
- **Seuil** : 0 % (strict).
- **Vérification** : Après le refactor, relancer une session de référence et extraire `min(mask_revive_latency_ms)` avec `jq '.probes.mask_revive_latency_ms.min'`. Attendu : >= 0 ms.
- **Anti-pattern interdit** : Si une valeur négative apparaît encore, **ne pas utiliser `abs()`** — chercher la cause racine (un champ mal synchronisé).

### Critère C2 — Cohérence inter-frame >= ~1 frame

- **Métrique** : `mask_revive_latency_ms` doit être >= `capture_period_ms` − pipeline_latency.
- **Vérification** : Sur une session avec détections slow à ~1 Hz, un mask LOST pendant ~1 seconde puis ré-acquis devrait produire une `mask_revive_latency_ms` dans la plage [500, 2500] ms (dérive du pipeline ~= 240 ms ± 100 ms). Des valeurs > 5000 ms ou < 0 ms indiquent un problème.
- **Outil** : `jq` sur les probes de session pour histogramme des percentiles.

### Critère C3 — Non-régression `mask_confirm_latency_ms`

- **Métrique** : `% de valeurs < 0`.
- **Seuil** : 0 %. Vérifié déjà avant le refactor (homogène capture-capture en L116).
- **Vérification** : Rejouer la session de référence post-refactor et confirmer `min >= 0`.

### Critère C4 — Non-régression `mask_lost_latency_ms`

- **Métrique** : `% de valeurs < 0`.
- **Seuil** : 0 % (post-refactor, si Q5-A).
- **Vérification** : Rejouer la session de référence post-refactor.

### Critère C5 — Les TTL s'écoulent toujours entre détections slow

- **Test manuel** : Avec `lost_after_s = 1.0`, après 1 frame sans détection, le mask ne doit PAS passer LOST. Après 1.1 seconde sans détection, il doit passer LOST.
- **Outil** : `bench.gauge("registry_lost")` doit rester à 0 pendant les rafales de détection, et augmenter après silence.
- **Vérification** : Observer `registry_lost_total` sur `bench_agg` d'une session avec alternance détection/silence.

### Critère C6 — Intégrité des gauges d'état

- `registry_confirmed`, `registry_pending`, `registry_lost` (registry.py L111-113) et `tracker_confirmed`, `tracker_pending`, `tracker_lost` (tracker.py L182-184) doivent être cohérents après modification de `tick_and_expire`.

---

## SECTION 6 — Ordre d'exécution / Phases

### Phase 1 — Répondre aux questions de la Section 1

| Question | Decision à prendre                           | Impact                                             |
| -------- | -------------------------------------------- | -------------------------------------------------- |
| Q1       | Q1-A, Q1-B ou Q1-C                           | Propagation de `detected_frame_ts` dans le tracker |
| Q2       | Confirmer la signature `apply_detections`    | Phase 3 tracker.py                                 |
| Q3       | Q3-A ou Q3-B                                 | Signature `mark_matched`                           |
| Q4       | Confirmer l'initialisation de `lost_after_s` | Phase 2 registry.py                                |
| Q5       | Q5-A ou Q5-B                                 | Comportement de `mask_lost_latency_ms`             |
| Q6       | Q6-A ou Q6-B                                 | Phase 5 documentation                              |
| Q7       | Q7-A, Q7-B ou Q7-C                           | Initialisation des champs `_frame_ts` dans Mask    |

**Délivrable Phase 0** : Document de décisions signé (soit intégré dans ce Plan_Timer.md, soit en réponse directe), validant une option par question.

---

### Phase 2 — core/mask.py

1. Ajouter les champs `last_seen_frame_ts` et `lost_since_frame_ts` dans le dataclass `Mask`.
2. Synchroniser dans `__post_init__`.
3. Corriger `mask_revive_latency_ms` (L119) : utiliser `lost_since_frame_ts`.
4. Ajouter la pose de `last_seen_frame_ts` et `lost_since_frame_ts` dans `transition()`.
5. Mettre à jour `to_dict()`.

**Critère de sortie** : pytest sur `Mask.transition()` avec mock `bench.probe` — les sondes latences émettent des valeurs positives.

---

### Phase 3 — tracker/registry.py

1. Ajouter `last_seen_frame_ts=ts` dans `registry.add()`.
2. Ajouter `last_seen_frame_ts=ts` dans `mark_matched()` (Q3).
3. Modifier la signature de `tick_and_expire()` pour recevoir `last_detected_frame_ts`.
4. Implémenter la sélection sémantique (i)/(ii) dans l'appel à `transition("missing")`.

**Critère de sortie** : Test d'intégration registry — création de mask, passage LOST sans détection, passage CONFIRMED : vérifier les champs `_frame_ts` sont cohérents.

---

### Phase 4 — tracker/tracker.py

1. Modifier la signature de `tick()` pour recevoir `last_detected_frame_ts`.
2. Transmettre à `registry.tick_and_expire()`.
3. Confirmer la signature de `apply_detections` (Q2).

**Critère de sortie** : `tracker.tick()` accepte un 3ᵉ paramètre optionnel `last_detected_frame_ts`.

---

### Phase 5 — main.py

1. Déclarer `last_detected_frame_ts = 0.0` avant la boucle.
2. Stocker `last_detected_frame_ts = detected_frame_ts` après `detector.get_result()`.
3. Passer `last_detected_frame_ts` à `tracker.tick()`.

**Critère de sortie** : La boucle principale compile et s'exécute sans erreur ; `last_detected_frame_ts` est mis à jour à chaque frame avec détection slow.

---

### Phase 6 — Documentation

1. Mettre à jour `Plan_Tracker.md` avec les changements.
2. (Option Q6-A) Mettre à jour `bench-probes.md` avec la base de temps capture pour les sondes latences.
3. Clôturer le ticket B-05a-revive avec référence à ce plan.
4. Mettre à jour B-05a-bis #4 si des dépendances sont identifiées.

---

## Annexe — État des champs Mask après implémentation

| Champ                 | Type              | Base           | Rôle                                                         |
| --------------------- | ----------------- | -------------- | ------------------------------------------------------------ |
| `last_seen_ts`        | `float`           | `perf_counter` | TTL : comparaison avec `now` pour détecter hors-vue          |
| `lost_since_ts`       | `Optional[float]` | `perf_counter` | TTL : début de la période LOST                               |
| `last_seen_frame_ts`  | `Optional[float]` | `capture`      | Latences : borne "dernière détection" pour les probes        |
| `lost_since_frame_ts` | `Optional[float]` | `capture`      | Latences : borne "entrée LOST" pour `mask_revive_latency_ms` |
| `created_ts`          | `float`           | `capture`      | Borne initiale pour `mask_confirm_latency_ms`                |

### Propagation de `detected_frame_ts` (flux de données)

```text
capturer.get_frame()       → frame_ts (capture, lag ~= 240ms)
        ↓
detector.get_result()     → detected_frame_ts
        ↓
last_detected_frame_ts    (stocké dans main.py entre frames)
        ↓
tracker.tick(..., last_detected_frame_ts)
        ↓
registry.tick_and_expire(..., last_detected_frame_ts)
        ↓
mask.transition("missing", last_detected_frame_ts)
        ↓
lost_since_ts = last_detected_frame_ts   (perf_counter)
lost_since_frame_ts = last_detected_frame_ts  (capture)
        ↓
mask.transition("matched", detected_frame_ts)
        ↓
lost_since_frame_ts (capture) → bench.probe("mask_revive_latency_ms", ...)
```
