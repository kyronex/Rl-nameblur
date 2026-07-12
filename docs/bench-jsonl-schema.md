# Schéma JSONL bench — Contrat normatif L0.4

## Sommaire

1. [Portée](#1-portée)
2. [Méta-champs communs](#2-méta-champs-communs)
   - 2.1 [Champs obligatoires](#21-méta-champs-obligatoires)
   - 2.2 [Règles d'évolution du `schema_version`](#22-règles-dévolution-du-schema_version)
3. [Canal `agg`](#3-canal-agg)
4. [Canal `fast`](#4-canal-fast)
5. [Canal `frame`](#5-canal-frame)
6. [Canal `events`](#6-canal-events)
7. [Canal `detections`](#7-canal-detections)
8. [Contrat des sections imbriquées](#8-contrat-des-sections-imbriquées)
   - 8.1 [`probes`](#81-probes)
   - 8.2 [`gauges`](#82-gauges)
   - 8.3 [`rates`](#83-rates)
   - 8.4 [`counts`](#84-counts)
9. [Matrice des sections par canal](#9-matrice-des-sections-par-canal)
10. [Référence d'implémentation](#10-référence-dimplémentation)
11. [Paramètres de configuration](#11-paramètres-de-configuration)

---

## 1. Portée

Ce document définit le schéma de chaque ligne JSONL émise par les writers bench (`bench/jsonl_writer.py`).
Il s'applique aux 5 canaux : `agg`, `fast`, `frame`, `events`, `detections`.
Le code est source de vérité ; ce document reflète l'implémentation au L0.4.

---

## 2. Méta-champs communs

### 2.1 Méta-champs obligatoires

Chaque ligne JSONL, tous canaux confondus, contient au minimum :

| Champ            | Type     | Description                                                                     |
| ---------------- | -------- | ------------------------------------------------------------------------------- |
| `schema_version` | `int`    | Version du schéma .                                                             |
| `ts`             | `float`  | Timestamp wall-clock UNIX (secondes,`time.time()`)                              |
| `mono`           | `float`  | Timestamp monotome (secondes, `time.perf_counter()`)                            |
| `session_id`     | `string` | Identifiant de session (format horodaté, ex : `20250612_085200`)                |
| `mode`           | `string` | Canal d'origine : `"frame"` / `"agg"` / `"fast"` / `"events"` / `"detections"`. |

### 2.2 Règles d'évolution du `schema_version`

- Le `schema_version` n'est incrémenté que lors d'un changement de contrat (nouveau champ obligatoire, suppression, changement de type).
- L'incrément se fait dans `BenchJsonlWriter._enqueue()` (bench/jsonl_writer.py).

---

## 3. Canal `agg`

### 3.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "agg",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "rates":  { "<rate_name>": <float>, ... }
}
```

### 3.2 Sections — présence

| Section  | Présence    | Condition                                           |
| -------- | ----------- | --------------------------------------------------- |
| `probes` | si non vide | au moins une sonde avec des mesures dans la fenêtre |
| `gauges` | si non vide | au moins une gauge posée                            |
| `rates`  | si non vide | au moins un count avec total > 0                    |

### 3.3 Sondes exclues

Le canal `agg` exclut :

- Les sondes préfixées par `fast_` (`_is_fast_probe` retourne True → exclues).
- Les sondes préfixées par `bench_writer_` (auto-sondes des writers).

---

## 4. Canal `fast`

### 4.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "fast",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "rates":  { "<rate_name>": <float>, ... }
}
```

### 4.2 Sections — présence

Identique à §3.2.

### 4.3 Sondes incluses / exclues

Le canal `fast` inclut **uniquement** les sondes préfixées `fast_` (via `_is_fast_probe`).
Toutes les autres sondes sont exclues.

**Note** : les sondes préfixées `F_*` (variantes exactes des probes fast) ne sont **pas** filtrées par `_is_fast_probe` — elles ne sont donc pas inclues dans le canal `fast`. Elles apparaissent dans le canal `frame` pour fournir des mesures exactes par frame.

---

## 5. Canal `frame`

### 5.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "frame",
  "probes": { "<probe_name>": <probe_stats>, ... },
  "gauges": { "<gauge_name>": <float>, ... },
  "counts": { "<count_name>": <int>, ... }
}
```

### 5.2 Sections — présence

Mêmes règles que §3.2.

### 5.3 Sémantique différentielle

- Les buffers `_frame_probes` et `_frame_counts` sont **vidés** après chaque appel à `snapshot_frame()`.
- `probes` contient les agrégats des mesures accumulées depuis le dernier snapshot.
- `counts` contient les totaux incrémentaux depuis le dernier snapshot.
- `gauges` reste un état instantané (pas de vidage).

### 5.4 Sondes exclues

Mêmes exclusions que `agg` : `fast_*` et `bench_writer_*`.
**Exception** : les sondes préfixées `F_*` ne sont pas filtrées et apparaissent dans ce canal (mesures exactes par frame).

---

## 6. Canal `events`

### 6.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "events",
  "events": {
    "records": [<LifecycleRecord>, ...]
  }
}
```

### 6.2 Section `events.records` — présence

Présente si et seulement si au moins un événement lifecycle a été émis depuis le dernier snapshot.
Le buffer est **drainé** après lecture.

### 6.3 LifecycleRecord — champs

| Champ                  | Type           | Description                                                                                                      |
| ---------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `event`                | `string`       | Valeur de `LifecycleEvent` : `CREATED` / `DETECTED` / `CONFIRMED` / `LOST` / `REVIVE` / `EXPIRED` / `EVICTED`    |
| `mask_id`              | `int`          | Identifiant unique du mask                                                                                       |
| `state`                | `string`       | État du mask au moment de l'événement : `PENDING` / `CONFIRMED` / `LOST` (jamais `EXPIRED`, `CREATED`, `REVIVE`) |
| `rx`, `ry`, `rw`, `rh` | `float`        | Géométrie du mask au moment de l'événement                                                                       |
| `confidence`           | `float`        | Score de confiance au moment de l'événement (0.0 – 1.0).                                                         |
| `created_ts`           | `float`        | Timestamp de création du mask (epoch, `time.perf_counter()`)                                                     |
| `event_ts`             | `float`        | Timestamp de l'événement (epoch, `time.perf_counter()`)                                                          |
| `total_matches_cumul`  | `int`          | Compteur cumulé de détections depuis la création                                                                 |
| `frames_matched`       | `int`          | Nombre de frames matchées au moment de l'événement                                                               |
| `source`               | `string\|null` | Dernière source de détection : `"slow"` / `"fast"`                                                               |
| `lost_since_ts`        | `float\|null`  | Timestamp d'entrée en état LOST (positionné pour CONFIRMED/LOST/REVIVE/EXPIRED ; null pour CREATED)              |
| `reason`               | `string\|null` | Raison optionnelle de l'événement (ex : `"ttl_expired"`, `"max_capacity"`)                                       |
| `revived`              | `bool\|null`   | `True` si événement `REVIVE` et mask en état `PENDING` ou `CONFIRMED` avant le revive ; `null` sinon             |
| `frame_id`             | `int`          | Identifiant de la frame au moment de l'événement                                                                 |
| `scores`               | `dict`         | Scores de matching au moment de l'événement (dict JSON-safe, peut être vide `{}`)                                |
| `hash_history`         | `list`         | Historique des phash du mask au moment de l'événement (liste de `int`, peut être vide `[]`)                      |

---

## 7. Canal `detections`

### 7.1 Structure

```json
{
  "schema_version": <int>,
  "ts": <float>,
  "mono": <float>,
  "session_id": <string>,
  "mode": "detections",
  "detections": {
    "records": [<DetectionRecord>, ...]
  }
}
```

### 7.2 Section `detections.records` — présence

Présente si et seulement si au moins une détection slow a été émise depuis le dernier snapshot.
Le buffer est **drainé** après lecture.

### 7.3 DetectionRecord — champs

| Champ                  | Type        | Description                                                               |
| ---------------------- | ----------- | ------------------------------------------------------------------------- |
| `frame_id`             | `int`       | Identifiant de la frame de détection                                      |
| `rx`, `ry`, `rw`, `rh` | `float`     | Coordonnées du rectangle de détection                                     |
| `phash`                | `int\|None` | Phash perceptuel de la crop (ou `null` si indisponible)                   |
| `source`               | `string`    | Source de la détection (actuellement `"slow"`)                            |
| `confidence`           | `float`     | Score de confiance de la détection                                        |
| `scores`               | `dict`      | Scores additionnels de la détection (dict JSON-safe, peut être vide `{}`) |

---

## 8. Contrat des sections imbriquées

### 8.1 `probes`

```json
"<probe_name>": {
  "avg":   <float>,
  "max":   <float>,
  "min":   <float>,
  "count": <int>
}
```

- `avg` = moyenne des valeurs sur la fenêtre (ou 0 si vide).
- `max` = maximum des valeurs sur la fenêtre.
- `min` = minimum des valeurs sur la fenêtre.
- `count` = nombre de mesures dans la fenêtre.

### 8.2 `gauges`

```json
"<gauge_name>": <float>
```

Valeur instantanée écrasante (pas de historique ni de vidage automatique).

### 8.3 `rates`

Débit moyen sur la fenêtre du canal.

```json
"<rate_name>": <float>
```

- Pour `agg` : `rate = total_count / agg.interval_s`.
- Pour `fast` : `rate = total_count / history_window_s`.

### 8.4 `counts`

Compteur différentiel sur la fenêtre entre deux appels à `snapshot_frame()`.

```json
"<count_name>": <int>
```

**Exclusivement sur le canal `frame`** : totaux différentiels depuis le dernier snapshot (buffer vidangé).

---

## 9. Matrice des sections par canal

| Section              | `frame` | `agg` | `fast` | `events` | `detections` |
| -------------------- | ------- | ----- | ------ | -------- | ------------ |
| `probes`             | ✅      | ✅    | ✅     | ❌       | ❌           |
| `gauges`             | ✅      | ✅    | ✅     | ❌       | ❌           |
| `counts`             | ✅      | ❌    | ❌     | ❌       | ❌           |
| `rates`              | ❌      | ✅    | ✅     | ❌       | ❌           |
| `events.records`     | ❌      | ❌    | ❌     | ✅       | ❌           |
| `detections.records` | ❌      | ❌    | ❌     | ❌       | ✅           |

Légende :

- ✅ = section autorisée (présente si non vide).
- ❌ = section non applicable à ce canal.

---

## 10. Référence d'implémentation

### Sources

| Fichier                 | Rôle                                                                    |
| ----------------------- | ----------------------------------------------------------------------- |
| `bench/bench.py`        | Registre central, snapshots, `emit_lifecycle`, `emit_detection`         |
| `bench/jsonl_writer.py` | Writers, validation par canal, rotation fichiers                        |
| `bench/lifecycle.py`    | Enum `LifecycleEvent`, TypedDict `LifecycleRecord` et `DetectionRecord` |

### Filtres de sondes

| Filtre             | Condition                          | Canal affecté                                   |
| ------------------ | ---------------------------------- | ----------------------------------------------- |
| `_is_fast_probe`   | `name.startswith("fast_")`         | `agg` (exclu), `frame` (exclu), `fast` (inclus) |
| `_is_writer_probe` | `name.startswith("bench_writer_")` | `agg` (exclu), `frame` (exclu), `fast` (non)    |
| `F_*` (variantes)  | `name.startswith("F_")`            | Non filtrées → canal `frame` uniquement         |

### Conventions d'auto-instrumentation

Les writers émettent des sondes automatiques préfixées `bench_writer_*` :

- `bench_writer_<mode>_queue_size` — taille courante de la queue (probe, ms, valeur = qsize)
- `bench_writer_<mode>_dropped` — nombre de lignes droppées (count)
- `bench_writer_<mode>_rejected_*` — lignes rejetées pour motif (count)

Ces sondes sont **toujours exclues** des snapshots métier ( `_is_writer_probe`).

---

## 11. Paramètres de configuration

### 11.1 Activation globale

`debug.bench.enabled` (`bool`) — master switch. Positionné dans `main.py` au boot.

### 11.2 Writers JSONL — paramètres communs

| Clé                                     | Type        | Default           | Description                                             |
| --------------------------------------- | ----------- | ----------------- | ------------------------------------------------------- |
| `debug.bench.writer.enabled`            | `bool`      | `false`           | Active les writers                                      |
| `debug.bench.writer.queue_maxsize`      | `int`       | `10000`           | Taille max de la queue du writer                        |
| `debug.bench.writer.session_id_format`  | `string`    | `"%Y%m%d_%H%M%S"` | Format horodaté du session_id                           |
| `debug.bench.writer.shutdown_timeout_s` | `float`     | `2.0`             | Timeout d'arrêt des threads                             |
| `debug.bench.writer.max_chars`          | `int\|null` | `null`            | Rotation fichier après N caractères (`null` = illimité) |
| `debug.bench.history_window_s`          | `float`     | `60.0`            | Fenêtre de rétention pour snapshots agg/fast            |

### 11.3 Canaux

Pour chaque canal `mode` ∈ {`agg`, `fast`, `frame`, `events`, `detections`} :

| Clé                             | Type     | Default                          | Description                                    |
| ------------------------------- | -------- | -------------------------------- | ---------------------------------------------- |
| `debug.bench.<mode>.enabled`    | `bool`   | `false`                          | Active le canal                                |
| `debug.bench.<mode>.path`       | `string` | `"logs/json/bench_<mode>.jsonl"` | Chemin du fichier de sortie                    |
| `debug.bench.<mode>.interval_s` | `float`  | `1.0`                            | Intervalle du snapshot (agg / fast uniquement) |

### 11.4 Frame Dumper

| Clé                                      | Type     | Default         | Description                                |
| ---------------------------------------- | -------- | --------------- | ------------------------------------------ |
| `debug.bench.frame_dumper.enabled`       | `bool`   | `false`         | Active le frame dumper                     |
| `debug.bench.frame_dumper.path`          | `string` | `"logs/frames"` | Répertoire de sortie                       |
| `debug.bench.frame_dumper.jpeg_quality`  | `int`    | `75`            | Qualité JPEG (0-100)                       |
| `debug.bench.frame_dumper.tail_frames`   | `int`    | `0`             | Nombre de frames après l'event àダMP (0-2) |
| `debug.bench.frame_dumper.ring_size`     | `int`    | `240`           | Taille du buffer circulaire de frames      |
| `debug.bench.frame_dumper.queue_maxsize` | `int`    | `256`           | Taille max de la queue                     |
