# Arbitrage Final — Cycle de Vie : 6 Événements

## 1. Arbitrage Final — 6 Événements du Lifecycle

| Événement     | Raison                                                         | Source                                                                               |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **CREATED**   | Naissance du mask, source (slow/fast), rect initial, confiance | `MaskRegistry.create()` — après `Mask(...)`                                          |
| **CONFIRMED** | Promotion PENDING→CONFIRMED, `frames_matched`, activation blur | `Mask.transition(event="matched", ...)` — quand `state` devient CONFIRMED            |
| **LOST**      | Première détection manquée, `lost_since_ts`, avant expiration  | `Mask.transition(event="missing", ...)` — quand `state` devient LOST                 |
| **REVIVE**    | LOST→CONFIRMED (re-détection), preuve robustesse tracker       | `Mask.transition(event="matched", ...)` — quand `state == LOST` + nouvelle détection |
| **EXPIRED**   | Purge finale, `total_matches_cumul` définitif, `duration_s`    | `MaskRegistry.tick_and_expire()` — après `del self._masks[uid]`                      |
| **EVICTED**   | Dépassement `max_masks`, mask supprimé (pire priority)         | `MaskRegistry._evict_one()` — après `del self._masks[worst.uid]`                     |

---

## 2. Schéma JSON Unifié — 6 Événements

### Enveloppe (valable pour tous les événements)

```json
{
  "schema_version": 1,
  "ts": 1234567.89,
  "mono": 1234567890123,
  "session_id": "cam-42",
  "mode": "lifecycle",
  "events": [
    {
      /* un record par événement émis */
    }
  ]
}
```

### Record `events[0]` — CREATED

```json
{
  "event": "CREATED",
  "mask_id": 1,
  "rx": 100.0,
  "ry": 150.0,
  "rw": 80.0,
  "rh": 120.0,
  "confidence": 1.0,
  "created_ts": 1234567.89,
  "event_ts": 1234567.89,
  "total_matches_cumul": 0,
  "source": "fast",
  "frames_matched": 1,
  "lost_since_ts": null,
  "state": "CONFIRMED",
  "reason": null,
  "revived": null
}
```

### Record `events[0]` — CONFIRMED

```json
{
  "event": "CONFIRMED",
  "mask_id": 1,
  "rx": 100.0,
  "ry": 150.0,
  "rw": 80.0,
  "rh": 120.0,
  "confidence": 1.0,
  "created_ts": 1234567.89,
  "event_ts": 1234567.93,
  "total_matches_cumul": 3,
  "source": null,
  "frames_matched": 3,
  "lost_since_ts": null,
  "state": "CONFIRMED",
  "reason": null,
  "revived": null
}
```

### Record `events[0]` — LOST

```json
{
  "event": "LOST",
  "mask_id": 1,
  "rx": 100.0,
  "ry": 150.0,
  "rw": 80.0,
  "rh": 120.0,
  "confidence": 1.0,
  "created_ts": 1234567.89,
  "event_ts": 1234568.1,
  "total_matches_cumul": 3,
  "source": null,
  "frames_matched": 0,
  "lost_since_ts": 1234568.1,
  "state": "LOST",
  "reason": null,
  "revived": null
}
```

### Record `events[0]` — REVIVE

```json
{
  "event": "REVIVE",
  "mask_id": 1,
  "rx": 105.0,
  "ry": 152.0,
  "rw": 78.0,
  "rh": 118.0,
  "confidence": 1.0,
  "created_ts": 1234567.89,
  "event_ts": 1234568.2,
  "total_matches_cumul": 4,
  "source": null,
  "frames_matched": 1,
  "lost_since_ts": null,
  "state": "CONFIRMED",
  "reason": null,
  "revived": true
}
```

### Record `events[0]` — EXPIRED

```json
{
  "event": "EXPIRED",
  "mask_id": 1,
  "rx": 105.0,
  "ry": 152.0,
  "rw": 78.0,
  "rh": 118.0,
  "confidence": 1.0,
  "created_ts": 1234567.89,
  "event_ts": 1234578.2,
  "total_matches_cumul": 4,
  "source": null,
  "frames_matched": 0,
  "lost_since_ts": 1234568.1,
  "state": "EXPIRED",
  "reason": "timeout_after_lost",
  "revived": null
}
```

### Record `events[0]` — EVICTED

```json
{
  "event": "EVICTED",
  "mask_id": 5,
  "rx": 300.0,
  "ry": 200.0,
  "rw": 60.0,
  "rh": 90.0,
  "confidence": 0.7,
  "created_ts": 1234567.0,
  "event_ts": 1234570.5,
  "total_matches_cumul": 1,
  "source": null,
  "frames_matched": 0,
  "lost_since_ts": 1234569.0,
  "state": "EVICTED",
  "reason": "max_masks_exceeded",
  "revived": null
}
```

---

## 3. Décorticage des champs par événement

| Champ                 |      CREATED      | CONFIRMED |   LOST    |   REVIVE    |          EXPIRED          |          EVICTED          |
| --------------------- | :---------------: | :-------: | :-------: | :---------: | :-----------------------: | :-----------------------: |
| `event`               |      CREATED      | CONFIRMED |   LOST    |   REVIVE    |          EXPIRED          |          EVICTED          |
| `mask_id`             |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `rx, ry, rw, rh`      |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `confidence`          |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `created_ts`          |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `event_ts`            |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `total_matches_cumul` |       **0**       |    ✅     |    ✅     |     ✅      |         ✅ final          |         ✅ final          |
| `source`              |  ✅ (slow/fast)   |    ❌     |    ❌     |     ❌      |            ❌             |            ❌             |
| `frames_matched`      |       **1**       |    ✅     |   **0**   |    **1**    |           **0**           |           **0**           |
| `lost_since_ts`       |        ❌         |    ❌     |    ✅     |     ❌      |            ✅             |            ✅             |
| `state`               | PENDING→CONFIRMED | CONFIRMED |   LOST    |  CONFIRMED  |          EXPIRED          |          EVICTED          |
| `reason`              |        ❌         |    ❌     |    ❌     |     ❌      | ✅ (`timeout_after_lost`) | ✅ (`max_masks_exceeded`) |
| `revived`             |        ❌         |    ❌     |    ❌     | ✅ (`true`) |            ❌             |            ❌             |
| `duration_s`          |        ❌         |    ❌     |    ❌     |     ❌      |            ✅             |            ✅             |
| `mode`                |     lifecycle     | lifecycle | lifecycle |  lifecycle  |         lifecycle         |         lifecycle         |
| `schema_version`      |         1         |     1     |     1     |      1      |             1             |             1             |
| `session_id`          |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `ts`                  |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |
| `mono`                |        ✅         |    ✅     |    ✅     |     ✅      |            ✅             |            ✅             |

---

## 4. Lieux d'Émission — Décision d'Architecture

> **Principe fondateur : `bench` est un global partagé.**
> `bench` est instancié une fois dans `bench.py` (`bench = BenchRegistry()`) et importé tel quel par tous les modules (`mask.py`, `registry.py`, `tracker.py`). Aucun module ne reçoit `bench` en injection de dépendance — tous y accèdent directement.

### Architecture d'émission

| Événement     | Méthode émettrice                  | Contexte                                     | Accès `bench`  |
| ------------- | ---------------------------------- | -------------------------------------------- | -------------- |
| **CREATED**   | `MaskRegistry.create()`            | Après `Mask(...)` + `_add()`                 | global `bench` |
| **CONFIRMED** | `Mask.transition(event="matched")` | Bloc `if state == PENDING … CONFIRMED`       | global `bench` |
| **LOST**      | `Mask.transition(event="missing")` | Bloc `elif event == 'missing': state = LOST` | global `bench` |
| **REVIVE**    | `Mask.transition(event="matched")` | Bloc `elif state == LOST … CONFIRMED`        | global `bench` |
| **EXPIRED**   | `MaskRegistry.tick_and_expire()`   | Après `del self._masks[uid]`                 | global `bench` |
| **EVICTED**   | `MaskRegistry._evict_one()`        | Après `del self._masks[worst.uid]`           | global `bench` |

### Justification des choix

**CONFIRMED / LOST / REVIVE → `Mask.transition()`**

Le code actuel de `mask.py` (lignes 111-137) montre que `Mask.transition()` contient déjà la logique de décision d'état, les probes de latence (`mask_confirm_latency_ms`, `mask_revive_latency_ms`, `mask_lost_latency_ms`) et les compteurs (`mask_promote_total`, `mask_revive_total`, `mask_to_lost_total`). Intégrer l'émission lifecycle **juste après** chaque transition d'état naturel (à l'intérieur des blocs `if state == ...`) est le plus cohérent :

```python
# mask.py — CONFIRMED émit depuis transition() (exemple)
if self.state == MaskState.PENDING and self.frames_matched >= self.confirm_after:
    self.state = MaskState.CONFIRMED
    bench.count('mask_promote_total')
    bench.probe('mask_confirm_latency_ms', ...)
    bench.event("CONFIRMED", mask=self, session_id=session_id, mode="lifecycle")
```

**CREATED → `MaskRegistry.create()` (pas `Mask.__init__`)**

`Mask` est un dataclass pur — son `__init__` ne connaît pas `session_id`, `mode`, ni ne doit porter la responsabilité d'émettre. C'est `registry.create()` qui a toute la connaissance : mask vient d'être créé, il est déjà enregistré, le `session_id` est connu. L'émission se fait **après** `Mask(...)` et `_add()`, avant le `return added` :

```python
# registry.py — CREATED émit depuis create()
added = self._add(mask)
bench.count("registry_create_total")
bench.event("CREATED", mask=mask, session_id=self._session_id, mode="lifecycle")
return added
```

**EXPIRED → `MaskRegistry.tick_and_expire()`**

Le code actuel (`registry.py`, ligne 109) fait `del self._masks[mask.uid]` après avoir ajouté `mask` à `expired`. L'émission se fait **après** la suppression du mask du registry :

```python
# registry.py — EXPIRED depuis tick_and_expire()
expired.append(mask)
del self._masks[mask.uid]
bench.count("registry_expire_total")
bench.event("EXPIRED", mask=mask, session_id=self._session_id, mode="lifecycle")
```

**EVICTED → `MaskRegistry._evict_one()`**

Même logique — après `del self._masks[worst.uid]` :

```python
# registry.py — EVICTED depuis _evict_one()
del self._masks[worst.uid]
bench.count("registry_evict_total")
bench.event("EVICTED", mask=worst, session_id=self._session_id, mode="lifecycle")
```

### Ce qui n'est PAS décidé ici (fichier 3 = bench.py)

L'implémentation de `bench.event()` et la sérialisation vers le fichier JSONL `lifecycle.jsonl` dépendent du fichier 3 (`bench.py`) qui apporte `"events"` dans `_VALID_MODES` / `_ALLOWED_SECTIONS`. **Cette section sera complétée après delivery du fichier 3.**

### Résumé des emplacements exacts dans le code

```text
mask.py
  Mask.transition(event="matched")
    ├─ CONFIRMED  →  après  self.state = MaskState.CONFIRMED (PENDING branch)
    └─ REVIVE     →  après  self.state = MaskState.CONFIRMED (LOST branch)
  Mask.transition(event="missing")
    └─ LOST       →  après  self.state = MaskState.LOST

registry.py
  MaskRegistry.create()
    └─ CREATED    →  après  _add() + bench.count("registry_create_total")
  MaskRegistry.tick_and_expire()
    └─ EXPIRED    →  après  del self._masks[mask.uid]
  MaskRegistry._evict_one()
    └─ EVICTED    →  après  del self._masks[worst.uid]
```

---

## 5. Glossaire des champs

| Champ                 | Type          | Définition                                                                            |
| --------------------- | ------------- | ------------------------------------------------------------------------------------- |
| `event`               | `str`         | Identifiant de l'événement : CREATED / CONFIRMED / LOST / REVIVE / EXPIRED / EVICTED  |
| `mask_id`             | `int`         | UID unique du mask (incrémenté par `MaskRegistry._next_uid`)                          |
| `rx, ry`              | `float`       | Coin supérieur-gauche du rectangle de détection                                       |
| `rw, rh`              | `float`       | Largeur / hauteur du rectangle                                                        |
| `confidence`          | `float`       | Score de confiance (0.0 – 1.0)                                                        |
| `created_ts`          | `float`       | Timestamp perf_counter de création du mask (`_frame_ts` à l'appel de `create()`)      |
| `event_ts`            | `float`       | Timestamp perf_counter de l'événementEmit                                             |
| `total_matches_cumul` | `int`         | Compteur cumulatif de toutes les détections (incrémenté sur chaque `event="matched"`) |
| `source`              | `str\|null`   | `"slow"` ou `"fast"` — **uniquement pour CREATED**                                    |
| `frames_matched`      | `int`         | Compteur de frames consécutives avec match (remis à 1 sur REVIVE, 0 sur LOST)         |
| `lost_since_ts`       | `float\|null` | Timestamp perf_counter du passage à LOST (null si non-LOST ou après REVIVE)           |
| `state`               | `str`         | État machine : PENDING / CONFIRMED / LOST / EXPIRED / EVICTED                         |
| `reason`              | `str\|null`   | Raison de fin : `timeout_after_lost` (EXPIRED), `max_masks_exceeded` (EVICTED)        |
| `revived`             | `bool\|null`  | `true` uniquement pour REVIVE                                                         |
| `session_id`          | `str`         | Identifiant de session (passé par le caller de `MaskRegistry`)                        |
| `mode`                | `str`         | Toujours `"events"`                                                                   |
| `ts`                  | `float`       | Timestamp perf_counter au moment de l'appel à `event()` (enveloppe)                   |
| `mono`                | `int`         | Monotonic raw counter (microsecondes, enveloppe)                                      |
| `schema_version`      | `int`         | Version du schéma : `1`                                                               |
